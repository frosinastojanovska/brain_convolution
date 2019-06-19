import os
import pickle
import numpy as np
import pandas as pd


class BrainDataset:
    def __init__(self, dataset_name, meta_data):
        self.dataset_name = dataset_name
        self.meta_data = meta_data
        self.ids = self.meta_data.network_name.values

    def get_brain_graph_and_class(self, subject_id):
        raise NotImplementedError

    def get_brain_class(self, subject_id):
        raise NotImplementedError


def read_matrix(file_path):
    """Reads adjacency matrix from given file paht to txt file

    :param file_path: file path to the adjacency path
    :type file_path: str
    :return: adjacency matrix
    :rtype: numpy.array
    """
    matrix = np.loadtxt(file_path)
    return matrix


def read_functional_connectomes_data():
    """Reads the meta data for the Functional Connectomes dataset

    :return: DataFrame with data information
    :rtype pandas.DataFrame
    """
    data = pd.read_csv('data/1000_Functional_Connectomes_info')[['upload_data.age_range_min',
                                                                 'upload_data.gender',
                                                                 'upload_data.subject_pool',
                                                                 'upload_data.network_name']]
    data.columns = ['age', 'gender', 'class', 'network_name']
    return data


def read_adhd_data():
    """Reads the meta data for the ADHD dataset

    :return: DataFrame with data information
    :rtype pandas.DataFrame
    """
    data = pd.read_csv('data/ADHD200_CC200_info')[['upload_data.age_range_min',
                                                   'upload_data.gender',
                                                   'upload_data.subject_pool',
                                                   'upload_data.network_name']]
    data.columns = ['age', 'gender', 'class', 'network_name']
    return data


def find_statistics(dir_path, stats_file):
    """Find the minimum and maximum values for the coordinates needed for normalization

    :param dir_path: path to the directory containing the dataset files
    :type dir_path: str
    :param stats_file: path where to save the calculated information
    :type stats_file: str
    :return: dictionary containing the maximum and minimum values for the coordinates
    :rtype: dict(str, numpy.array)
    """
    if os.path.exists(stats_file):
        with open(stats_file, 'rb') as f:
            return pickle.load(f)
    subject_files = os.listdir(dir_path)
    subject_files = [x for x in subject_files if 'region_xyz_centers' in x]
    max_vals = [0, 0, 0]
    min_vals = [0, 0, 0]
    for file in subject_files:
        data = np.loadtxt(dir_path + file)
        maxs = data.max(axis=0).tolist()
        max_vals = [max(x, y) for x, y in zip(max_vals, maxs)]
        mins = data.min(axis=0).tolist()
        min_vals = [min(x, y) for x, y in zip(min_vals, mins)]

    stats = {'max': np.array(max_vals),
             'min': np.array(min_vals)}

    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
    return stats


def data_generator(dataset, config, shuffle=True):
    """ A generator that returns brain graphs and corresponding target class.

    :param dataset: the BrainDataset object to pick data from
    :type dataset: BrainDataset object
    :param config: the configuration object
    :type config: Config object
    :param shuffle: if True, shuffles the samples before every epoch
    :type shuffle: bool
    :return: returns a Python generator. Upon calling next() on it, the generator
            returns two lists, inputs and outputs
    :rtype: Python generator
    """
    b = 0  # batch item index
    subject_index = -1
    subject_ids = np.array(dataset.subject_ids)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next protein. Shuffle if at the start of an epoch.
            subject_index = (subject_index + 1) % len(subject_ids)
            if shuffle and subject_index == 0:
                np.random.shuffle(subject_ids)

            # Get protein sequence and GT classes.
            subject_id = subject_ids[subject_index]
            matrix, node_features, classes = dataset.get_brain_graph_and_class(subject_id)

            # Init batch arrays
            if b == 0:
                batch_adj = []
                batch_nodes = []
                batch_classes = []

            # Add to batch
            batch_nodes.append(node_features)
            batch_adj.append(matrix)
            batch_classes.append(classes)
            b += 1

            # Batch full?
            if b >= config.BATCH_SIZE:
                batch_nodes = np.array(batch_nodes)
                batch_adj = np.array(batch_adj)
                batch_classes = np.array(batch_classes)
                yield [batch_nodes, batch_adj], batch_classes

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
