import numpy as np
from typing import List, Tuple

import torch
from torch_geometric.data import Data
from torch import Tensor
from utils.dataloader_utils import build_dataset


class DiffMatrix:
    def __init__(self, ratio):
        self.ratio = ratio
        self.difference_matrix = None

    def compute(self, train_dataset: List[Data], train_labels: Tensor):
        negative_matrix = np.zeros_like(train_dataset[0].adj)
        negative_count = 1e-6
        positive_matrix = np.zeros_like(train_dataset[0].adj)
        positive_count = 1e-6
        for i in range(len(train_labels)):
            if train_labels[i] != 1:  # HIV dataset label only contain -1 and 1
                negative_matrix = np.add(negative_matrix, train_dataset[i].adj)
                negative_count += 1
            elif train_labels[i] == 1:
                positive_matrix = np.add(positive_matrix, train_dataset[i].adj)
                positive_count += 1
        difference_matrix: Tensor = np.abs(positive_matrix / positive_count - negative_matrix / negative_count)
        # max_val = np.max(difference_matrix)
        num_elements = int(torch.numel(difference_matrix) * self.ratio)
        threshold = torch.topk(torch.flatten(difference_matrix), num_elements).values[num_elements - 1]
        difference_matrix[difference_matrix > threshold] = 1
        difference_matrix[difference_matrix != 1] = 0
        self.difference_matrix = difference_matrix
        return self

    def apply(self, dataset, args, y):
        dataset_matrix = torch.zeros((len(dataset), dataset[0].adj.shape[0], dataset[0].adj.shape[1]))
        for i in range(len(dataset)):
            dataset_matrix[i] = dataset[i].adj * self.difference_matrix
        return build_dataset(dataset_matrix, args, y)