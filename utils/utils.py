import csv
import math
import os
from typing import List

import numpy
import scipy
import torch
import random
import numpy as np
from torch import Tensor
import shutil
from typing import Optional


def split_dataset(dataset, split_mode, *args, **kwargs):
    assert split_mode in ['rand', 'ogb', 'wikics', 'preload']
    if split_mode == 'rand':
        assert 'train_ratio' in kwargs and 'test_ratio' in kwargs
        train_ratio = kwargs['train_ratio']
        test_ratio = kwargs['test_ratio']
        num_samples = dataset.x.size(0)
        train_size = int(num_samples * train_ratio)
        test_size = int(num_samples * test_ratio)
        indices = torch.randperm(num_samples)
        return {
            'train': indices[:train_size],
            'val': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        }
    elif split_mode == 'ogb':
        return dataset.get_idx_split()
    elif split_mode == 'wikics':
        assert 'split_idx' in kwargs
        split_idx = kwargs['split_idx']
        return {
            'train': dataset.train_mask[:, split_idx],
            'test': dataset.test_mask,
            'val': dataset.val_mask[:, split_idx]
        }
    elif split_mode == 'preload':
        assert 'preload_split' in kwargs
        assert kwargs['preload_split'] is not None
        train_mask, test_mask, val_mask = kwargs['preload_split']
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }


def seed_everything(seed):
    print(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # set random seed for numpy
    # set deterministic for conv in cudnn
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True 
    torch.manual_seed(seed) # set random seed for CPU
    torch.cuda.manual_seed_all(seed) # set random seed for all GPUs


def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())


def read_csv_to_list(path) -> List[List[str]]:
    try:
        filename = open(path, newline='')
        list_generated = list(csv.reader(filename))
    except Exception as e:
        print(e)
        return []
    return list_generated


def save_list_to_csv(list_given, path):
    with open(path, 'wb') as file:
        writer = csv.writer(file)
        writer.writerows(list_given)


def edges2adj(edges: Tensor, num_nodes: int = 0) -> numpy.ndarray:
    if num_nodes == 0:
        num_nodes = int(math.sqrt(edges.shape[0]))
    adj = numpy.zeros((num_nodes, num_nodes))
    for index, edge in enumerate(edges):
        adj[index % num_nodes, int(index / num_nodes)] = edge
    return adj


def save_edges_to_mat(edges: numpy.ndarray, filename):
    edge_dict = {'edges': edges}
    scipy.io.savemat(filename, edge_dict)


def edge_index_to_adj_matrix(edge_index: Tensor, edge_attr: Tensor, num_node: int) -> numpy.ndarray:
    adj = numpy.zeros((num_node, num_node))
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        adj[source, target] = edge_attr[i].item()
    return adj


def archive_files(source, dst):
    existing_figs = os.listdir(source)
    for fig in existing_figs:
        if not os.path.isdir(source + fig):
            if os.path.exists(dst + fig):
                os.remove(dst + fig)
            shutil.move(os.path.abspath(source + fig),
                        os.path.abspath(dst + fig))


def mkdir_if_needed(folder: str):
    if not os.path.exists(folder):
        os.mkdir(folder)


def mkdirs_if_needed(folders: List[str]):
    for folder in folders:
        mkdir_if_needed(folder)


## common model
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))


def _remove_self_loops(edge_index, edge_attr: torch.Tensor, edge_flags: torch.Tensor):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return edge_index, edge_attr[mask], edge_flags[mask]


def _add_self_loops(edge_index, edge_weight: Optional[torch.Tensor] = None,
                    edge_flags: Optional[torch.Tensor] = None,
                    fill_value: float = 1., num_nodes: Optional[int] = None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, self-loops will be added with edge weights
    denoted by :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (float, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((N,), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    if edge_flags is not None:
        assert edge_flags.numel() == edge_index.size(1)
        loop_weight = edge_flags.new_full((N,), fill_value)
        edge_flags = torch.cat([edge_flags, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight, edge_flags