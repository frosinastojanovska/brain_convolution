import tensorflow as tf
import keras.backend as K
import keras.models as KM
import keras.layers as KL

from model.layers import GraphConv, ReadoutLayer


class BrainConvolutionModel:
    """
    BrainConvolution model.
    """

    def __init__(self, config):
        """
        :param config: config object containing the configuration parameters
        :type: Config object
        """
        self.config = config

    def build(self):
        """ Build BrainConvolution architecture.
        :return: keras model
        :rtype: KM.Model
        """
        x = KL.Input(shape=[None, self.config.NODE_DIM],
                     name='input_node_features', dtype=tf.float32)

        adj = KL.Input(shape=[None, None],
                       name='input_adjacency_matrix', dtype=tf.float32)

        encoded_state = self.graph_encoder(x, adj)

        dense_out = KL.Dense(128, name='dense_layer1')(encoded_state)
        dense_out = KL.BatchNormalization(axis=-1)(dense_out)
        dense_out = KL.ReLU()(dense_out)
        dense_out = KL.Dense(64, name='dense_layer2')(dense_out)
        dense_out = KL.BatchNormalization(axis=-1)(dense_out)
        dense_out = KL.ReLU()(dense_out)
        output = KL.Dense(self.config.NUM_CLASSES, activation='softmax', name=f'dense_layer3')(dense_out)

        return KM.Model([x, adj], output, name='brainconvolution')

    def graph_encoder(self, x, adj):
        """Encode graph into vector representation with graph convolution network

        :param x: hidden representation of nodes
        :param adj: adjacency matrix including the edge features
        :return: latent representation of graph
        """
        # Fist graph convolution layer
        x1 = GraphConv(self.config.DIM_CONV1)([x, adj])
        x1 = KL.BatchNormalization(axis=-1)(x1)
        x1 = KL.ReLU()(x1)

        h = ReadoutLayer(self.config.DIM_READOUT)(x1)
        h = KL.BatchNormalization(axis=-1)(h)
        h = KL.ReLU()(h)

        return h
