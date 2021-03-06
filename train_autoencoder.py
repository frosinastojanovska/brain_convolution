import os
import keras
import random
import numpy as np
from keras.optimizers import Adam

from model.config import Config
from model.autoencoder_model import BrainAutoencoderModel
from model.utils import BrainDataset, read_matrix, read_functional_connectomes_data, data_generator, find_statistics

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class FunctionalConnectomes(BrainDataset):
    def __init__(self, dataset_name, subject_ids, meta_data, stats):
        super().__init__(dataset_name, meta_data)
        self.subject_ids = subject_ids
        self.stats = stats

    def get_brain_graph_and_class(self, subject_id):
        """Returns the brain graph and class for the given subject id

        :param subject_id: id of the subject
        :type subject_id: str
        :return: adjacency matrix of the brain graph, node features, encoded class
        :rtype: numpy.array, numpy.array, numpy.array
        """
        matrix = read_matrix(f'data/1000_Functional_Connectomes/{subject_id}_connectivity_matrix_file.txt')
        matrix[matrix < 0] = 0
        coordinates = np.loadtxt(f'data/1000_Functional_Connectomes/{subject_id}_region_xyz_centers_file.txt')
        node_features = (coordinates - self.stats['min']) / (self.stats['max'] - self.stats['min'])

        return matrix, node_features, matrix

    def get_brain_class(self, subject_id):
        """Returns the class for the given subject id

        :param subject_id: id of the subject
        :type subject_id: str
        :return: encoded class
        :rtype: numpy.array
        """
        matrix = read_matrix(f'data/1000_Functional_Connectomes/{subject_id}_connectivity_matrix_file.txt')
        matrix[matrix < 0] = 0
        return matrix


def train_validation_split(subject_ids, split=0.1):
    """ Split list of subject ids into train and test list
    :param subject_ids: list of subject ids
    :type subject_ids: list(str)
    :param split: ratio value for validation data
    :type split: float
    :return: train subject ids, validation subject ids
    :rtype: list(str), list(str)
    """
    size = len(subject_ids)
    split_index = int(round(size * (1 - split)))
    random.shuffle(subject_ids)
    train_ids = subject_ids[:split_index]
    validation_ids = subject_ids[split_index:]
    return train_ids, validation_ids


if __name__ == '__main__':
    data = read_functional_connectomes_data()
    stats = find_statistics('data/1000_Functional_Connectomes/', 'data/1000_functional_connectomes_stats.pkl')
    subject_ids = data.network_name.values.tolist()
    subject_graph_files = os.listdir('data/1000_Functional_Connectomes/')
    subject_ids = [subject for subject in subject_ids if
                   subject + '_connectivity_matrix_file.txt' in subject_graph_files]
    train_subject_ids, val_subject_ids = train_validation_split(subject_ids, 0.1)
    train_dataset = FunctionalConnectomes('1000_functional_connectomes', train_subject_ids,
                                          data.loc[data['network_name'].isin(train_subject_ids)], stats)
    validation_dataset = FunctionalConnectomes('1000_functional_connectomes', val_subject_ids,
                                               data.loc[data['network_name'].isin(val_subject_ids)], stats)
    config = Config(node_dim=3, num_classes=0, batch_size=2)

    train_data_generator = data_generator(train_dataset, config, shuffle=True)
    val_data_generator = data_generator(validation_dataset, config, shuffle=True)

    model = BrainAutoencoderModel(config).build()

    model_filepath = 'trained/model/autoencoder-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'trained/logs/autoencoder.log'

    if not os.path.exists(os.path.dirname(logs_filepath)):
        os.makedirs(os.path.dirname(logs_filepath))

    if not os.path.exists(os.path.dirname(model_filepath)):
        os.makedirs(os.path.dirname(model_filepath))

    checkpoint = keras.callbacks.ModelCheckpoint(model_filepath, verbose=1, save_weights_only=True, mode='min')
    csv_logger = keras.callbacks.CSVLogger(logs_filepath)

    model.summary()
    optimizer = Adam(0.001, amsgrad=True, decay=config.WEIGHT_DECAY)
    model.compile(loss='mse', optimizer=optimizer)
    model.fit_generator(train_data_generator, 10, 2000, callbacks=[checkpoint, csv_logger],
                        validation_data=val_data_generator, validation_steps=len(val_subject_ids) // config.BATCH_SIZE)
