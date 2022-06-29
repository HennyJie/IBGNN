import glob
from sklearn import metrics
import csv
from typing import List, Dict
from utils.load_node_labels import load_cluster_info_from_txt
from utils.utils import read_csv_to_list


class SystemCount:
    def __init__(self):
        self.negative = dict()
        self.positive = dict()


def calculate_dominant_communities(dataset: str):
    mapping_path = 'datasets/' + dataset + '_Community.csv'
    system_mapping = read_csv_to_list(mapping_path)
    metrics_folder = 'metrics/'
    metrics_name = ['strength', 'degree', 'betweenness', 'clustering_coef']
    for view in ['dti', 'fmri']:
        output_name = f'dominant_communities_{view}.csv'
        output_file = open(metrics_folder + output_name, 'w', newline='')
        output_writer = csv.writer(output_file, delimiter=',')
        output_writer.writerow(['Metrics name', 'Top systems'])

        for metrics_path in metrics_name:
            with open(metrics_folder + metrics_path + '.csv', newline='') as metrics_file:
                # for each metrics, make a dictionary covering count for each sample
                system_count = SystemCount()

                print(f'Now reading {metrics_path}.csv...')
                metrics_file_reader = csv.reader(metrics_file)
                count_systems(metrics_file_reader, system_count, system_mapping, view)

                positive_systems = sorted(system_count.positive.items(), key=lambda item: item[1], reverse=True)
                print('------Positive----')
                print(positive_systems)
                rank_of_positive_systems = [x[0] for x in positive_systems]
                print(' '.join(rank_of_positive_systems))
                # Write top 3 systems to output
                output_writer.writerow([metrics_path + ', positive', ' '.join(rank_of_positive_systems[0:3])])

                negative_systems = sorted(system_count.negative.items(), key=lambda item: item[1], reverse=True)
                print('------Negative----')
                print(negative_systems)
                rank_of_negative_systems = [x[0] for x in negative_systems]
                print(' '.join(rank_of_negative_systems))
                # Write top 3 systems to output
                output_writer.writerow([metrics_path + ', negative', ' '.join(rank_of_negative_systems[0:3])])


def count_systems(metrics_file_reader, system_count, system_mapping, view):
    for row in metrics_file_reader:
        sample_name = row[0]
        # Skip original, unfiltered explain graph and graph from the other view.
        if 'explainer' in sample_name and 'full' not in sample_name and view in sample_name:
            for order in range(1, 6):  # top 5 nodes in each graph
                node_index = int(row[order])
                # Mapping file row structure: Index (0), Abbreviation (1), System (2)
                system = system_mapping[node_index][2]
                if 'positive' in sample_name:
                    increment_count(system_count.positive, system)
                elif 'negative' in sample_name:
                    increment_count(system_count.negative, system)
                else:
                    assert False, "The csv file contains neither negative nor positive."


def increment_count(current_count: Dict, system):
    if system in current_count:
        current_count[system] += 1
    else:
        current_count[system] = 1


def calculate_metrics():
    with open('metrics.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['filename', 'rand_score', 'adjusted_rand_score',
                         'adjusted_mutual_info_score', 'completeness_score',
                         'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score',
                         'v_measure_score'])

        for filename in glob.glob("modularity/*.csv"):
            with open(filename, 'r') as file:
                print(filename)
                dataset_name = 'HIV' if 'HIV' in filename else 'BP'
                txt_name = "datasets/New_Node_AAL90.txt" if dataset_name == 'HIV' else "datasets/New_Node_Brodmann82.txt"
                labels_true = load_cluster_info_from_txt(txt_name)
                labels_pred_str = file.readlines()
                labels_pred: List[int] = list(map(int, labels_pred_str))
                print(labels_true)
                print(labels_pred)
                rand_score = metrics.rand_score(labels_true, labels_pred)
                print(f"rand_score: {rand_score:.4f}")

                adjusted_rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
                print(f"adjusted_rand_score: {adjusted_rand_score:.4f}")

                adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
                print(f"adjusted_mutual_info_score: {adjusted_mutual_info_score:.4f}")

                completeness_score = metrics.completeness_score(labels_true, labels_pred)
                print(f"completeness_score: {completeness_score:.4f}")

                fowlkes_mallows_score = metrics.fowlkes_mallows_score(labels_true, labels_pred)
                print(f"fowlkes_mallows_score: {fowlkes_mallows_score:.4f}")

                homogeneity_score = metrics.homogeneity_score(labels_true, labels_pred)
                print(f"homogeneity_score: {homogeneity_score:.4f}")

                mutual_info_score = metrics.mutual_info_score(labels_true, labels_pred)
                print(f"mutual_info_score: {mutual_info_score:.4f}")

                v_measure_score = metrics.v_measure_score(labels_true, labels_pred)
                print(f"v_measure_score: {v_measure_score:.4f}")
                # silhouette_score = metrics.silhouette_score(labels_true, labels_pred)
                # print(f"silhouette_score: {silhouette_score:.4f}")

                writer.writerow([filename, f"{rand_score:.4f}", f"{adjusted_rand_score:.4f}",
                                 f"{adjusted_mutual_info_score:.4f}", f"{completeness_score:.4f}",
                                 f"{fowlkes_mallows_score:.4f}", f"{homogeneity_score:.4f}",
                                 f"{mutual_info_score:.4f}", f"{v_measure_score:.4f}"])


if __name__ == '__main__':
    calculate_metrics()
    calculate_dominant_communities('BP')
