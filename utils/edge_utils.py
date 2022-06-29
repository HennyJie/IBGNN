from typing import List

import numpy
import torch
from torch import Tensor

from models.paired_data import PairedData


def generate_full_edges(num_nodes) -> Tensor:
    full_edge_index = numpy.zeros((2, num_nodes * num_nodes), dtype=numpy.long)

    for source in range(0, num_nodes):
        for target in range(0, num_nodes):
            row = source * num_nodes + target
            full_edge_index[0, row] = source
            full_edge_index[1, row] = target

    full_edge_index_tensor = torch.LongTensor(full_edge_index)
    return full_edge_index_tensor


def map_edges_attr(attr: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
    new_edge_attrs = numpy.zeros((num_nodes * num_nodes,))
    for i in range(attr.shape[0]):
        source = edge_index[0, i]
        target = edge_index[1, i]

        # maps edge attr to new index
        new_index = source * num_nodes + target
        new_edge_attrs[new_index] = attr[i].item()

    return Tensor(new_edge_attrs)


def generate_community_labels_for_edges(edge_index: Tensor, node_labels: List[int]) -> Tensor:
    edge_count = edge_index.shape[1]
    edge_community_label = numpy.zeros(edge_count)
    for row in range(edge_count):
        source: Tensor = edge_index[0, row]
        target: Tensor = edge_index[1, row]
        if node_labels[source.item()] == node_labels[target.item()]:
            # If source.label == target.label,
            # then set the corresponding community label to 1
            edge_community_label[row] = 1

    return Tensor(edge_community_label)