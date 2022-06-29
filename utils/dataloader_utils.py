import networkx as nx
from typing import Union

import numpy
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch.utils.data import random_split
from models.paired_data import PairedData
import numpy as np
from utils.maskable_list import MaskableList
# newly added dependency for tensor decomposition
import tensorly as tl
from tensorly.decomposition import parafac, tucker  # cp decomposition and tucker decomposition
from scipy.io import savemat
from numpy import linalg as LA


# for degree_bin node features
def binning(a, n_bins=10):
    n_graphs = a.shape[0]
    n_nodes = a.shape[1]
    _, bins = np.histogram(a, n_bins)
    binned = np.digitize(a, bins)
    binned = binned.reshape(-1, 1)
    enc = OneHotEncoder()
    return enc.fit_transform(binned).toarray().reshape(n_graphs, n_nodes, -1).astype(np.float32)


# for LDP node features
def LDP(g, key='deg'):
    x = np.zeros([len(g.nodes()), 5])

    deg_dict = dict(nx.degree(g))
    for n in g.nodes():
        g.nodes[n][key] = deg_dict[n]

    for i in g.nodes():
        nodes = g[i].keys()

        nbrs_deg = [g.nodes[j][key] for j in nodes]

        if len(nbrs_deg) != 0:
            x[i] = [
                np.mean(nbrs_deg),
                np.min(nbrs_deg),
                np.max(nbrs_deg),
                np.std(nbrs_deg),
                np.sum(nbrs_deg)
            ]

    return x


# mark the edge index in tiled n^2 vector
def generate_edge_flag(num_nodes, edge_index):
    edge_flag = np.full((num_nodes ** 2,), False)
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i]
        target = edge_index[1, i]
        new_index = source * num_nodes + target
        edge_flag[new_index] = True
    # print(edge_flag.shape)
    return edge_flag


# node coordinates, node labels and region names
def load_txt(file: str) -> [[Union[float, str]]]:
    if file is None:
        return None
    array_from_txt = list()
    with open(file, 'r') as f:
        for row in f.readlines():
            split = row.split(sep='\t')
            try:
                processed_row: [Union[float, str]] = [float(x) for x in split[0:3]]  # coordinates
                processed_row.append(int(split[3]))  # node label (cluster)
                processed_row.append(split[5])  # node region name
                array_from_txt.append(processed_row)
            except:  # This skips headers and other illegal rows
                continue
        return array_from_txt  # len(array_from_txt): num_nodes


# node labels
def load_cluster_info_from_txt(file: str) -> [int]:
    if file is None:
        return None
    cluster_info = list()
    node_info_array = load_txt(file)
    for row in node_info_array:
        cluster_info.append(int(row[3]))
    return cluster_info


def save_dataset_to_mat(dataset: numpy.ndarray, y: numpy.ndarray,  name: str):
    x = numpy.empty((dataset.shape[0], 1), dtype=object)
    dti = numpy.zeros((dataset.shape[0], 1))
    for i in range(dataset.shape[0]):
        x[i, 0] = dataset[i].reshape(
            (dataset[i].shape[0], dataset[i].shape[1], 1)
        )
    savemat(f'../datasets/{name}.mat', {'X': x, 'label': y.reshape((y.shape[0], 1)),
                                     'dti': dataset.swapaxes(0, 2)})


# for other matlab shallow baselines, not related to our IBGNN design
def load_data_shallow_baseline(dataset_name, path):
    m = loadmat(f'{path}/{dataset_name}.mat')
    if dataset_name == 'PPMI' or dataset_name == 'PPMI_balanced':
        data = m['X'] if dataset_name == 'PPMI' else m['X_new']
        a1 = np.zeros((data.shape[0], 2, 84, 84))
        for (index, sample) in enumerate(data):
            mapping = [0, 2]
            for view_index in range(2):
                # Assign the first view in the three views of PPMI to a1
                a1[index, view_index, :, :] = sample[0][:, :, mapping[view_index]]
    elif dataset_name in ['HIV', 'BP']:
        num_views = 2
        data = [m['fmri'], m['dti']]
        x_1 = data[0][:, :, 0]
        a1 = np.zeros((data[0].shape[2], num_views, x_1.shape[0], x_1.shape[1]))
        for view in range(num_views):
            for sample_index in range(data[0].shape[2]):
                a1[sample_index, view, :, :] = data[view][:, :, sample_index]
    else:
        raise AssertionError("Invalid dataset name")

    labels = m['label'] if dataset_name != 'PPMI_balanced' else m['label_new']
    labels = np.array(labels, dtype=int).reshape((len(labels),))
    labels[labels == -1] = 0

    return a1, labels


# for m2e and mic matlab shallow baselines, not related to our IBGNN design
def load_features(feature: str, dataset_name: str, path: str):
    m = loadmat(f'{path}/{feature}_{dataset_name}.mat')
    if feature == 'm2e':
        feature_mat = m['centroidF']
    elif feature == 'mic':
        feature_mat = m['A']
    else:
        raise ValueError(f'Feature {feature} not found.')
    return feature_mat


# construct knn graph on A (optional)
def extract_knn(a: numpy.ndarray, k: int):
    if k == 0:
        return a
    processed_a = numpy.zeros(a.shape)
    for sample_index, graph in enumerate(a):
        k_indexes = numpy.argsort(graph)[:, -1:-k - 1:-1]
        for row_index, row in enumerate(graph):
            processed_a[sample_index, row_index, k_indexes[row_index]] = row[k_indexes[row_index]]
    return processed_a


# load adjacency matrix A from raw matlab
def load_data_singleview(args, path, modality: str, node_labels):
    dataset = args.dataset_name
    m = loadmat(f'{path}/{dataset}.mat')

    labels = m['label'] if dataset != 'PPMI_balanced' else m['label_new']
    y = torch.Tensor(labels).long().flatten()
    y[y == -1] = 0

    if dataset == 'PPMI' or dataset == 'PPMI_balanced':
        data = m['X'] if dataset == 'PPMI' else m['X_new']
        a1 = np.zeros((data.shape[0], 84, 84))
        if modality == 'dti':
            model_index = 2
        else:
            model_index = int(modality)

        for (index, sample) in enumerate(data):
            # Assign the first view in the three views of PPMI to a1
            a1[index, :, :] = sample[0][:, :, model_index]
    else:
        a1 = m[modality].transpose((2, 1, 0))

    # for sample in a1:
    #     nonzero = numpy.count_nonzero(sample)
    #     print(f'Non-zero: {nonzero} out of {sample.size}. percentage {nonzero/sample.size*100:.2f}')

    # when args.top_k == 0, return the original a1
    # a1 = extract_knn(a1, args.top_k)
    a1 = torch.Tensor(a1)

    # preprocess the adjancency matrix A with shallow dimension reduction
    if args.shallow == "cp":  # cp tensor decomposition
        factors = parafac(tl.tensor(a1), rank=args.rank)
        a1_transformed = tl.cp_to_tensor(factors)
        a1 = torch.from_numpy(a1_transformed)
    elif args.shallow == "tucker":  # tucker tensor decomposition
        core, factors = tucker(tl.tensor(a1), rank=[args.rank_dim0, args.rank_dim1, args.rank_dim2])
        a1_transformed = tl.tucker_to_tensor((core, factors))
        a1 = torch.from_numpy(a1_transformed)

    bin_edges, data_list = build_dataset(a1, args, y)

    return data_list, bin_edges, y


def build_dataset(a1, args, y):
    x1 = compute_x(a1, args)
    data_list = MaskableList([])
    all_edge_weights_list = []
    for i in range(a1.shape[0]):
        edge_index, edge_attr = dense_to_sparse(a1[i])
        edge_flag = generate_edge_flag(x1.shape[1], edge_index)
        data = Data(x=x1[i], edge_index=edge_index, edge_attr=edge_attr, y=y[i], adj=a1[i], edge_flag=edge_flag)
        data_list.append(data)
        single_graph_edge_weights = [weight for weight in edge_attr.numpy()]
        all_edge_weights_list.extend(single_graph_edge_weights)
    # edge weights distribution
    hist, bin_edges = np.histogram(all_edge_weights_list, bins=10)
    return bin_edges, data_list


def compute_x(a1, args):
    # construct node features X
    if args.node_features == 'identity':
        x = torch.cat([torch.diag(torch.ones(a1.shape[1]))] * a1.shape[0]).reshape([a1.shape[0], a1.shape[1], -1])
        x1 = x.clone()

    elif args.node_features == 'node2vec':
        X = np.load(f'./{args.dataset_name}_{args.modality}.emb', allow_pickle=True).astype(np.float32)
        x1 = torch.from_numpy(X)

    elif args.node_features == 'degree':
        a1b = (a1 != 0).float()
        x1 = a1b.sum(dim=2, keepdim=True)

    elif args.node_features == 'degree_bin':
        a1b = (a1 != 0).float()
        x1 = binning(a1b.sum(dim=2))

    elif args.node_features == 'adj': # edge profile
        x1 = a1.float()

    elif args.node_features == 'LDP': # degree profile
        a1b = (a1 != 0).float()
        x1 = []
        n_graphs: int = a1.shape[0]
        for i in range(n_graphs):
            x1.append(LDP(nx.from_numpy_array(a1b[i].numpy())))

    elif args.node_features == 'eigen':
        _, x = LA.eig(a1.numpy())

    x1 = torch.Tensor(x1).float()
    return x1


def random_split_dataset(data_list):
    n_graphs = len(data_list)
    train_cnt = int(n_graphs * 0.8)
    test_cnt = n_graphs - train_cnt
    train, test = random_split(data_list, [train_cnt, test_cnt])
    train_mask = torch.zeros(n_graphs).bool()  # Should mask be twice as long?
    test_mask = train_mask.clone()
    train_mask[train.indices] = True
    test_mask[test.indices] = True
    return train, test
