import numpy
from pandas import DataFrame
from pandas import read_csv
import numpy as np
import nibabel as nib
import csv

metrics_folder = "../metrics/"
csv_names = ['betweenness', 'clustering_coef', 'degree', 'strength']
dataset_folder = "../datasets/"


def find_popular_nodes() -> (numpy.ndarray, numpy.ndarray):
    """
    Finds the most popular nodes in the network.
    """
    positives = []
    negatives = []
    for csv_name in csv_names:
        df = read_csv(metrics_folder + csv_name + '.csv')

        for row in df.iterrows():
            if 'positive' in row[1][0]:
                positives.append(row[1][1:6].array.to_numpy())
            else:
                negatives.append(row[1][1:6].array.to_numpy())

    # count the numbers of each node and sort them
    positives, negatives = numpy.array(positives, dtype=np.int32).flatten(), \
                           numpy.array(negatives, dtype=np.int32).flatten()
    # Count the number of times each node appears in the positive and negative arrays
    positive_counts = np.bincount(positives)
    negative_counts = np.bincount(negatives)
    # find the top 5 nodes
    top_5_positives = np.argsort(positive_counts)[-5:]
    top_5_negatives = np.argsort(negative_counts)[-5:]
    return top_5_positives, top_5_negatives


def map_nodes(nodes: numpy.ndarray, dataset_name) -> list:
    if 'brodmann' in dataset_name:
        mapping = list(csv.reader(open(dataset_folder + 'BP_mapping.csv', 'r', encoding='utf-8')))
    elif 'ppmi' in dataset_name:
        mapping = list(csv.reader(open(dataset_folder + 'PPMI_mapping.csv', 'r', encoding='utf-8')))
    else:
        return nodes.tolist()

    new_nodes = []
    for node in nodes:
        new_nodes.append(int(mapping[node-1][1]))

    return new_nodes


def create_nii(nodes: numpy.ndarray, template_name, output_name) -> None:
    nii_obj: nib.Nifti1Image = nib.load(dataset_folder + template_name)
    # noinspection PyTypeChecker
    data = numpy.array(nii_obj.get_fdata())

    new_nodes = map_nodes(nodes, template_name)
    mapping = dict()
    for i, node in enumerate(new_nodes):
        mapping[node] = nodes[i]

    # Set values to zero if they are not in the nodes array
    # Tried to vectorize this, but it didn't work
    # sum_of_count = 0
    # for i in range(84):
    #     count = np.sum(data == i)
    #     sum_of_count += count
    #     print(f'Node ID: {i}, Count of positions in nni: {count}')
    #
    # print(f'Total number of positions in nii: {sum_of_count}')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if data[i, j, k] not in new_nodes:
                    data[i, j, k] = 0
                else:
                    data[i, j, k] = mapping[int(data[i, j, k])]
    new_image = nib.Nifti1Image(data, nii_obj.affine, nii_obj.header)
    nib.save(new_image, output_name)


if __name__ == '__main__':
    positives, negatives = find_popular_nodes()
    print(f'Positives: {positives}, \nNegatives: {negatives}')
    # create_nii(positives, 'brodmann.nii', 'brodmann_positives_mask.nii')
    create_nii(negatives, 'brodmann.nii', 'brodmann_negatives_mask.nii')
    print('Done!')

