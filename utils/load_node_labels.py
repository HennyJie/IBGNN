from typing import Union


def load_txt(file: str) -> [[Union[float, str]]]:
    array_from_txt = list()
    with open(file, 'r') as f:
        for row in f.readlines():
            split = row.split(sep='\t')
            try:
                processed_row: [Union[float, str]] = [float(x) for x in split[0:3]]
                processed_row.append(int(split[3]))
                processed_row.append(split[5])
                array_from_txt.append(processed_row)
            except:  # This skips headers and other illegal rows
                continue
        return array_from_txt


def load_cluster_info_from_txt(file: str) -> [int]:
    if file is None:
        return None
    cluster_info = list()
    graph_info_array = load_txt(file)
    for row in graph_info_array:
        cluster_info.append(int(row[3]))
    return cluster_info


def load_roi_info_from_txt(file: str) -> [int]:
    roi_info = list()
    graph_info_array = load_txt(file)
    for row in graph_info_array:
        roi_info.append(row[4].replace('\n', ''))
    return roi_info
