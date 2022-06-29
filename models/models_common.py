import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter, Linear
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn import global_add_pool, global_mean_pool
from models.message_passing.message_passing import ModifiedMessagePassing as MessagePassing
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from utils.utils import maybe_num_nodes, _remove_self_loops, _add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


def gcn_norm(edge_index, edge_flag, edge_attr=None, num_nodes=None, improved=False,
             do_add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if do_add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1),), dtype=dtype,
                                   device=edge_index.device)

        if do_add_self_loops:
            if isinstance(edge_index, Tensor):
                if isinstance(edge_flag, Tensor):
                    edge_index, edge_attr, edge_flag = _remove_self_loops(edge_index, edge_attr, edge_flag)
                    edge_index, edge_attr, edge_flag = _add_self_loops(edge_index, edge_attr, edge_flag,
                                                                       num_nodes=num_nodes)
                else:
                    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                    edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_attr, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_attr * deg_inv_sqrt[col], edge_flag


class MPConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True, bias: bool = True):
        super(MPConv, self).__init__(aggr='add')

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.__explain__ = False

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.lin = torch.nn.Linear(out_channels*2 + 1, out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_attr, edge_flag):
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight, edge_flag = gcn_norm(  # yapf: disable
                        edge_index, edge_flag, edge_attr, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight, edge_flag)
                else:
                    edge_index, edge_weight, edge_flag = cache[0], cache[1], cache[2]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_attr, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, edge_flag, x=x, edge_attr=edge_weight)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr.view(-1, 1)], dim=1)
        return self.lin(msg)


class IBGConv(torch.nn.Module):
    def __init__(self, input_dim, args, num_classes):
        super(IBGConv, self).__init__()
        self.activation = torch.nn.ReLU()
        self.convs = torch.nn.ModuleList()

        hidden_dim = args.hidden_dim
        num_layers = args.n_GNN_layers
        self.pooling = args.pooling

        for i in range(num_layers):
            if i == 0:
                conv = MPConv(input_dim, hidden_dim)
            elif i != num_layers - 1:
                conv = MPConv(hidden_dim, hidden_dim)
            else:
                conv = MPConv(hidden_dim, num_classes)
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_attr, edge_flag, batch):
        z = x
        edge_attr[edge_attr < 0] = - edge_attr[edge_attr < 0]
        for i, conv in enumerate(self.convs):
            z = conv(z, edge_index, edge_attr, edge_flag)
            if i != len(self.convs) - 1:
                z = F.relu(z)  # [N * M, F]
                z = F.dropout(z, training=self.training)
            if self.pooling == 'sum':
                g = global_add_pool(z, batch)  # [N, F]
            elif self.pooling == 'mean':
                g = global_mean_pool(z, batch)  # [N, F]
            else:
                raise NotImplementedError('Pooling method not implemented')

        return F.log_softmax(g, dim=-1)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, n_classes=0):
        super(MLP, self).__init__()
        self.net = []
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(activation())
        for _ in range(num_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(activation())
        self.net = torch.nn.Sequential(*self.net)
        self.shortcut = torch.nn.Linear(input_dim, hidden_dim)

        if n_classes != 0:
            self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out = self.net(x) + self.shortcut(x)
        if hasattr(self, 'classifier'):
            return out, self.classifier(out)
        return out
