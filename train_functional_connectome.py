import os
import keras
import random
import numpy as np
from keras.optimizers import Adam

from model.config import Config
from model.brain_convolution_model import BrainConvolutionModel
from model.utils import BrainDataset, read_matrix, read_functional_connectomes_data, data_generator

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class FunctionalConnectomes(BrainDataset):
    def __init__(self, dataset_name, subject_ids, meta_data):
        super().__init__(dataset_name, meta_data)
        self.subject_ids = subject_ids
        self.class_mappings = {'Female': np.array([1, 0]), 'Male': np.array([0, 1])}

    def get_brain_graph_and_class(self, subject_id):
        matrix = read_matrix(f'data/1000_Functional_Connectomes/{subject_id}_connectivity_matrix_file.txt')
        matrix = np.expand_dims(matrix, axis=-1)
        node_features = np.identity(matrix.shape[0])
        class_id = self.meta_data.loc[self.meta_data['network_name'] == subject_id]['gender'].values[0]
        encoded_class = self.class_mappings[class_id]
        return matrix, node_features, encoded_class

    def get_brain_class(self, subject_id):
        class_id = self.meta_data.loc[self.meta_data['network_name'] == subject_id]['gender'].values[0]
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
    data = read_functional_connectomes_data()
    subject_ids = data.network_name.values.tolist()
    train_subject_ids, val_subject_ids = train_validation_split(subject_ids, 0.1)
    train_dataset = FunctionalConnectomes('1000_functional_connectomes', train_subject_ids,
                                          data.loc[data['network_name'].isin(train_subject_ids)])
    validation_dataset = FunctionalConnectomes('1000_functional_connectomes', val_subject_ids,
                                               data.loc[data['network_name'].isin(val_subject_ids)])
    config = Config(177, num_classes=2, batch_size=1)

    train_data_generator = data_generator(train_dataset, config, shuffle=True)
    val_data_generator = data_generator(validation_dataset, config, shuffle=True)

    model = BrainConvolutionModel(config).build()

    model_filepath = 'trained/model/functional_connectomes_brain_convolution-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'trained/logs/functional_connectomes.log'

    if not os.path.exists(os.path.dirname(logs_filepath)):
        os.makedirs(os.path.dirname(logs_filepath))

    if not os.path.exists(os.path.dirname(model_filepath)):
        os.makedirs(os.path.dirname(model_filepath))

    checkpoint = keras.callbacks.ModelCheckpoint(model_filepath, verbose=1, save_weights_only=True, mode='min')
    csv_logger = keras.callbacks.CSVLogger(logs_filepath)

    model.summary()
    optimizer = Adam(0.001, amsgrad=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit_generator(train_data_generator, 20, 100, validation_data=val_data_generator,
                        validation_steps=len(val_subject_ids) // config.BATCH_SIZE)
