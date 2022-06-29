import numpy

from utils.utils import read_csv_to_list
from typing import List


def take_system_name(elem):
    return elem[2]


def generate_system_ordered_adj(dataset: str, adj: numpy.ndarray) -> (numpy.ndarray, List[int]):
    mapping, system_mapping = get_system_mapping(dataset)

    # print(mapping)

    new_adj = numpy.zeros(adj.shape)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            new_adj[mapping[i], mapping[j]] = adj[i, j]

    return new_adj


def get_divider_lines(dataset_name: str):
    mapping, system_mapping = get_system_mapping(dataset_name)
    # get divider lines
    divider_lines: List[int] = list()
    divider_lines.append(0)
    names = ['' for _ in range(len(mapping))]
    current_system = system_mapping[0][2]
    start_index = 0
    for i in range(len(system_mapping)):
        if system_mapping[i][2] != current_system:
            divider_lines.append(i)
            names[int((i + start_index) / 2)] = current_system
            current_system = system_mapping[i][2]
            start_index = i
    names[int((len(system_mapping) + start_index) / 2)] = current_system
    divider_lines.append(len(system_mapping))
    return divider_lines, names


def get_system_mapping(dataset):
    system_mapping: List[List[str]] = read_csv_to_list(f"datasets/{dataset}_Community.csv")
    system_mapping = system_mapping[1:]
    system_mapping.sort(key=take_system_name)
    # print(system_mapping)
    mapping = list()
    # Now: [original index, abbr, system]
    for index, line in enumerate(system_mapping):
        # New index, original index
        mapping.append(int(line[0]) - 1)
    return mapping, system_mapping


def plot_heatmap(adj: numpy.ndarray, dataset_name: str, filename: str):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # sns.heatmap(adj, cmap='Reds', vmin=0, vmax=1, cbar=False)
    numpy.fill_diagonal(adj, 0)
    divider_lines, names = get_divider_lines(dataset_name)
    plt.figsize = (7, 7)
    ax = sns.heatmap(adj, cmap='bwr', vmin=0, vmax=1, cbar=False, xticklabels=names, yticklabels=names.copy())
    ax.hlines(divider_lines, *ax.get_xlim(), colors='black', linestyles='dashed')
    ax.vlines(divider_lines, *ax.get_xlim(), colors='black', linestyles='dashed')
    ax.set_aspect('equal', adjustable='box')

    plt.savefig(f"fig/{filename}_heatmap.png")
