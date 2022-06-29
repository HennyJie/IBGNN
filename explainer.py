from typing import Optional, Union
import matplotlib.axes
import torch
from tqdm import tqdm
import networkx as nx
from utils.edge_utils import *
from models.message_passing.message_passing import ModifiedMessagePassing as MessagePassing
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
from torch import Tensor
from utils.utils import edges2adj, save_edges_to_mat, mkdir_if_needed, edge_index_to_adj_matrix
from analysis.generate_heatmap import generate_system_ordered_adj, plot_heatmap
from matplotlib.pyplot import figure
import torch.nn.functional as F

EPS = 1e-15


class GNNExplainer(torch.nn.Module):
    r"""Modified GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        return_type (str, optional): Denotes the type of output from
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model returns
            the logarithm of probabilities), :obj:`"prob"` (the model returns
            probabilities) and :obj:`"raw"` (the model returns raw scores).
            (default: :obj:`"log_prob"`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'community_regularizer': 0.01,
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.01,
                 return_type: str = 'log_prob',
                 log: bool = True, labels: List[int] = None, num_clusters=6, remove_loss=None):
        super(GNNExplainer, self).__init__()
        if remove_loss is None:
            remove_loss = list()
        self.remove_loss = remove_loss
        assert return_type in ['log_prob', 'prob', 'raw']
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.return_type = return_type
        self.log = log
        self.node_labels = labels  # node labels
        self.edge_community_label: Optional[Tensor] = None
        self.num_clusters = num_clusters

    def __set_masks__(self, x, edge_index, init="normal"):
        num_features = x.size(1)

        (N, F), E = x.size(), edge_index.size(1)

        self.node_feat_mask = torch.nn.Parameter(torch.ones(F))
        self.edge_mask = torch.nn.Parameter(torch.ones(E))

        self.divergence_u = torch.nn.Parameter(torch.randn((self.num_clusters, num_features)))
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __graphloss__(self, out, pred_label, y):
        # supervised cross_entropy term
        class_loss = F.nll_loss(out, y) * 10

        # maximize the agreement between the prediction \hat{y} on the original graph and 
        # \hat{y}^{\prime} on explanation graph induced by the mask
        mask_loss = F.nll_loss(out, pred_label) * 10

        m = self.edge_mask.sigmoid()
        # sparsity loss
        if 'sparsity' in self.remove_loss:
            sparse_loss = 0
        else:
            sparse_loss = self.coeffs['edge_size'] * m.sum() / 10

        # element-wise entropy loss
        if 'entropy' in self.remove_loss:
            entropy_loss = 0
        else:
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            entropy_loss = self.coeffs['edge_ent'] * ent.mean()

        total_loss = class_loss + mask_loss + sparse_loss + entropy_loss
        return total_loss

    def build_Q(self, x, divergence_u):
        catted = [1, self.num_clusters, 1]
        Z = torch.unsqueeze(x, 1).repeat(catted)
        Q = torch.pow(torch.sum(torch.sum((Z - divergence_u) ** 2, dim=2), 1), -1)
        return Q / torch.sum(Q, dim=0, keepdim=True)

    def target_distribution(self, q: Tensor) -> Tensor:
        numerator = (q ** 2) / torch.sum(q, 0)
        p = (numerator.t() / torch.sum(numerator, 0)).t()
        return p

    def divergence_loss(self, x, divergence_u):
        divergence_q = self.build_Q(x, divergence_u)
        divergence_p = self.target_distribution(q=divergence_q)
        divergence_loss = torch.sum(divergence_p * torch.log(torch.div(divergence_p, divergence_q)))
        return divergence_loss

    def explainer_train(self, train_iterator: DataLoader, device, args):
        # self.model.eval()
        num_nodes = next(iter(train_iterator)).adj.shape[0]

        full_edges: Tensor = generate_full_edges(num_nodes)
        full_edges = full_edges.to(device)

        if self.node_labels is not None:
            self.edge_community_label = generate_community_labels_for_edges(full_edges, self.node_labels).to(device)
        explainer_train_loader = self.generate_explainer_loader(train_iterator, full_edges, device, num_nodes,
                                                                batch_size=args.train_batch_size)
        self.model.train()

        self.__set_masks__(next(iter(train_iterator)).x, full_edges, args.train_batch_size)
        self.to(device)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explainer Training')

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        data: Data
        for epoch in range(1, self.epochs + 1):
            for data in explainer_train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                pruned_edge_mask = self.prune_edge_mask(self.edge_mask, data.edge_flag).to(device)
                explained_edge_attr = data.edge_attr * pruned_edge_mask.view(1, -1).sigmoid()
                explained_edge_attr = explained_edge_attr.squeeze()
                data.edge_attr = explained_edge_attr
                data.edge_flag = pruned_edge_mask
                out = self.model(data)

                loss = self.__graphloss__(out, pred_label=data.pred, y=data.y)
                loss.sum().backward()
                optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)

        print(f"(Explainer Train) | Epoch={epoch:03d}, loss={loss.item():.4f}")

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask

    @torch.no_grad()
    def generate_explainer_loader(self, iterator: DataLoader, full_edges: Tensor, device,
                                  num_nodes: int, batch_size: int = 1) -> DataLoader:
        """

        Note: The input iterator dataloader must be of batch size 1

        """
        self.model.eval()
        original_batch_size = iterator.batch_size
        assert original_batch_size == 1
        new_dataset = list()
        data: Data
        for data_index, data in enumerate(iterator):
            data = data.to(device)
            output = self.model(data)
            pred = output.max(dim=1)[1]

            data = data.to('cpu')
            new_edge_attrs = self.map_edge_attrs(data)  # edge weights in the full graph

            if self.node_labels is None:
                community_label = None
            else:
                community_label = generate_community_labels_for_edges(edge_index=full_edges,
                                                                      node_labels=self.node_labels)

            new_data = Data(x=data.x, edge_index=data.edge_index, full_edge_index=full_edges,
                            edge_attr=data.edge_attr, new_edge_attr=new_edge_attrs, edge_flag=data.edge_flag,
                            y=data.y, pos=data.pos, community_label=community_label, pred=pred)
            new_dataset.append(new_data)

        explainer_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)
        return explainer_loader

    @staticmethod
    def map_edge_attrs(data: Data) -> Tensor:
        return torch.reshape(data.adj, (-1,))

    def prune_edge_mask(self, edge_mask: Tensor, edge_flag: Tensor) -> Tensor:
        catted_edge_mask = torch.cat(len(edge_flag) * [edge_mask])
        if (len(edge_flag) != 1):
            edge_flag = [i[0] for i in edge_flag]
            edge_flag = numpy.concatenate(edge_flag)
        pruned_edge_mask = catted_edge_mask[edge_flag]
        return pruned_edge_mask

    def mask_dataloader(self, node_feat_mask, edge_mask, iterator, args, node_atts, device, batch_size):
        mask_adj = edges2adj(edge_mask)
        # save_edges_to_mat(mask_adj, f'result/{args.dataset_name}_mask.mat')

        masked_data = []
        for i, data in enumerate(iterator):
            data = data.to(device)
            pruned_edge_mask = self.prune_edge_mask(edge_mask=edge_mask, edge_flag=data.edge_flag).to(device)
            pruned_edge_attr = data.edge_attr * pruned_edge_mask
            new_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=pruned_edge_attr,
                            y=data.y, edge_mask=pruned_edge_mask, edge_flag=data.edge_flag)
            masked_data.append(new_data)


        masked_loader = DataLoader(masked_data, batch_size=batch_size, shuffle=True)

        return masked_loader

    def plot_explanations(self, data, filtered_edges, dataset_name, seed, node_feat_mask, node_atts, index=0):
        positivity = 'positive' if data.y.item() == 1 else 'negative'
        num_nodes = data.x.shape[0]
        figure(figsize=(8, 6), dpi=300)
        G, edges, unfiltered_edges = self.visualize_graph(data.edge_index,
                                                          data.edge_attr,
                                                          x=node_feat_mask,
                                                          y=torch.FloatTensor(
                                                              self.node_labels),
                                                          node_atts=node_atts,
                                                          threshold_num=300)
        edges = generate_system_ordered_adj(dataset_name, edges)
        unfiltered_edges = generate_system_ordered_adj(dataset_name, unfiltered_edges)
        # print(f"Filtered Nonzero: {numpy.count_nonzero(edges)}, Full nonzero: {numpy.count_nonzero(unfiltered_edges)}")
        mkdir_if_needed('result')
        save_edges_to_mat(unfiltered_edges,
                          f"./fig/explainer_{dataset_name}_seed{seed}_full_{positivity}_{index}.mat")
        plot_heatmap(edges, dataset_name, f"explained_{dataset_name}_seed{seed}_{positivity}_{index}")
        # plt.savefig(f"./fig/{filtered_edges}%_graph_explainer_graph_{dataset_name}_{index}.jpg")
        # plt.show()

        # save node/edge
        numpy.savetxt(f"./fig/explainer_{dataset_name}_seed{seed}_{positivity}_{index}.edge", edges, delimiter='\t')
        save_edges_to_mat(edges, f"./fig/explainer_{dataset_name}_seed{seed}_filtered_{positivity}_{index}.mat")

    def visualize_graph(self, edge_index, edge_attr: Optional[Tensor],
                        node_atts: Optional[List[List[Union[float, str]]]], x=None,
                        y: Optional[torch.FloatTensor] = None, threshold_num=None) \
            -> (matplotlib.axes.Axes, Optional[numpy.ndarray]):
        r"""Visualizes the graph given an edge mask
        :attr:`edge_mask`.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        if edge_attr is not None:
            assert edge_attr.size(0) == edge_index.size(1)

        subset = torch.arange(edge_index.max().item() + 1, device=edge_index.device)

        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1),
                                   device=edge_index.device)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y.cpu()
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index, att=edge_attr, x=x, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        att_array = numpy.array([data['att'] for _, _, data in G.edges(data=True)])
        min_att, max_att = numpy.amin(att_array), numpy.amax(att_array)
        # reward = (max_att - min_att) / 10
        # att_array = self.reward_edge_postprocessing(att_array, edge_index, reward)
        # range_att = max_att - min_att
        # if range_att == 0:
        #     range_att = max_att
        graph_nodes = G.nodes

        edges = edge_index_to_adj_matrix(edge_index, edge_attr, y.shape[0])

        unfiltered_edges = edges.copy()
        if threshold_num is not None:
            edges = self.denoise_graph(edges, 0, threshold_num=threshold_num)

        return G, edges, unfiltered_edges

    def denoise_graph(self, adj, node_idx, feat=None, label=None, threshold=None, threshold_num=None,
                      max_component=True):
        """Cleaning a graph by thresholding its node values.

        Args:
            - adj               :  Adjacency matrix.
            - node_idx          :  Index of node to highlight (TODO What is this used for??)
            - feat              :  An array of node features.
            - label             :  A list of node labels.
            - threshold         :  The weight threshold.
            - theshold_num      :  The maximum number of nodes to threshold.
            - max_component     :  TODO  Looks like this has already been implemented
        """
        num_nodes = adj.shape[-1]
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.nodes[node_idx]["self"] = 1
        if feat is not None:
            for node in G.nodes():
                G.nodes[node]["feat"] = feat[node]
        if label is not None:
            for node in G.nodes():
                G.nodes[node]["label"] = label[node]

        if threshold_num is not None:
            # this is for symmetric graphs: edges are repeated twice in adj
            adj_threshold_num = threshold_num * 2
            # adj += np.random.rand(adj.shape[0], adj.shape[1]) * 1e-4
            neigh_size = len(adj[adj > 0])
            threshold_num = min(neigh_size, adj_threshold_num)
            threshold = numpy.sort(adj[adj > 0])[-threshold_num]

        if threshold is not None:
            weighted_edge_list = [
                (i, j, adj[i, j] if adj[i, j] >= threshold else 0)
                for i in range(num_nodes)
                for j in range(num_nodes)
            ]
        else:
            weighted_edge_list = [
                (i, j, adj[i, j])
                for i in range(num_nodes)
                for j in range(num_nodes)
                if adj[i, j] > 1e-6
            ]
        G.add_weighted_edges_from(weighted_edge_list)
        # if max_component:
        #     largest_cc = max(nx.connected_components(G), key=len)
        #     G = G.subgraph(largest_cc).copy()
        # else:
        #     # remove zero degree nodes
        #     G.remove_nodes_from(list(nx.isolates(G)))
        # adj_matrix = networkx.linalg.graphmatrix.adjacency_matrix(G)
        for i in range(num_nodes):
            for j in range(num_nodes):
                adj[i][j] = weighted_edge_list[i * num_nodes + j][2]
        return adj
        # return weighted_edge_list

    def reward_edge_postprocessing(self, edge_att: numpy.ndarray, edge_index: Tensor, reward):
        community_label = generate_community_labels_for_edges(edge_index=edge_index, node_labels=self.node_labels)
        return Tensor(edge_att) + reward * community_label