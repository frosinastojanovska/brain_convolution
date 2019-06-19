import os
import keras
import random
import numpy as np
from keras.optimizers import Adam

from model.config import Config
from model.brain_convolution_model import BrainConvolutionModel
from model.utils import BrainDataset, read_matrix, read_adhd_data, data_generator, find_statistics

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class ADHD(BrainDataset):
    def __init__(self, dataset_name, subject_ids, meta_data, stats):
        super().__init__(dataset_name, meta_data)
        self.subject_ids = subject_ids
        self.stats = stats
        self.class_mappings = {'ADHD-Combined': np.array([0, 0, 0, 1]),
                               'ADHD-Hyperactive/Impulsive': np.array([0, 0, 1, 0]),
                               'ADHD-Inattentive': np.array([0, 1, 0, 0]),
                               'Typically Developing': np.array([1, 0, 0, 0])}

    def get_brain_graph_and_class(self, subject_id):
        """Returns the brain graph and class for the given subject id

        :param subject_id: id of the subject
        :type subject_id: str
        :return: adjacency matrix of the brain graph, node features, encoded class
        :rtype: numpy.array, numpy.array, numpy.array
        """
        matrix = read_matrix(f'data/ADHD200_CC200/{subject_id}_connectivity_matrix_file.txt')
        matrix[matrix < 0] = 0
        matrix = np.expand_dims(matrix, axis=-1)
        coordinates = np.loadtxt(f'data/ADHD200_CC200/{subject_id}_region_xyz_centers_file.txt')
        node_features = (coordinates - self.stats['min']) / (self.stats['max'] - self.stats['min'])
        class_id = self.meta_data.loc[self.meta_data['network_name'] == subject_id]['class'].values[0]
        encoded_class = self.class_mappings[class_id]
        return matrix, node_features, encoded_class

    def get_brain_class(self, subject_id):
        """Returns the class for the given subject id

        :param subject_id: id of the subject
        :type subject_id: str
        :return: encoded class
        :rtype: numpy.array
        """
        class_id = self.meta_data.loc[self.meta_data['network_name'] == subject_id]['class'].values[0]
        encoded_class = self.class_mappings[class_id]
        return encoded_class


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
    data = read_adhd_data()
    stats = find_statistics('data/ADHD200_CC200/', 'data/adhd_stats.pkl')
    subject_ids = data.network_name.values.tolist()
    subject_graph_files = os.listdir('data/ADHD200_CC200/')
    subject_ids = [subject for subject in subject_ids if
                   subject + '_connectivity_matrix_file.txt' in subject_graph_files]
    train_subject_ids, val_subject_ids = train_validation_split(subject_ids, 0.1)
    train_dataset = ADHD('ADHD200_CC200', train_subject_ids,
                         data.loc[data['network_name'].isin(train_subject_ids)], stats)
    validation_dataset = ADHD('ADHD200_CC200', val_subject_ids,
                              data.loc[data['network_name'].isin(val_subject_ids)], stats)
    config = Config(node_dim=3, num_classes=4, batch_size=3)

    train_data_generator = data_generator(train_dataset, config, shuffle=True)
    val_data_generator = data_generator(validation_dataset, config, shuffle=True)

    model = BrainConvolutionModel(config).build()

    model_filepath = 'trained/model/adhd_brain_convolution-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'trained/logs/adhd.log'

    if not os.path.exists(os.path.dirname(logs_filepath)):
        os.makedirs(os.path.dirname(logs_filepath))

    if not os.path.exists(os.path.dirname(model_filepath)):
        os.makedirs(os.path.dirname(model_filepath))

    checkpoint = keras.callbacks.ModelCheckpoint(model_filepath, verbose=1, save_weights_only=True, mode='min')
    csv_logger = keras.callbacks.CSVLogger(logs_filepath)

    model.summary()
    optimizer = Adam(0.001, amsgrad=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit_generator(train_data_generator, 10, 20, callbacks=[checkpoint, csv_logger],
                        validation_data=val_data_generator, validation_steps=len(val_subject_ids) // config.BATCH_SIZE)
