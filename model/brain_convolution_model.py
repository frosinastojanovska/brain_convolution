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
        x = KL.Input(batch_shape=[self.config.BATCH_SIZE, None, self.config.NODE_DIM],
                     name='input_node_features', dtype=tf.float32)

        adj = KL.Input(batch_shape=[self.config.BATCH_SIZE, None, None],
                       name='input_adjacency_matrix', dtype=tf.float32)

        encoded_state = self.graph_encoder(x, adj)

        dense_out = KL.Dense(128, name='dense_layer1')(encoded_state)
        dense_out = KL.BatchNormalization(axis=-1)(dense_out)
        dense_out = KL.LeakyReLU()(dense_out)
        dense_out = KL.Dropout(0.1)(dense_out)
        dense_out = KL.Dense(64, name='dense_layer2')(dense_out)
        dense_out = KL.BatchNormalization(axis=-1)(dense_out)
        dense_out = KL.LeakyReLU()(dense_out)
        dense_out = KL.Dropout(0.1)(dense_out)
        output = KL.Dense(self.config.NUM_CLASSES, activation='softmax', name='dense_layer3')(dense_out)

        return KM.Model([x, adj], output, name='brainconvolution')

    def graph_encoder(self, x, adj, use_read_out=False):
        """Encode graph into vector representation with graph convolution network

        :param x: hidden representation of nodes
        :param adj: adjacency matrix including the edge features
        :param use_read_out: flag to use read out operation
        :return: latent representation of graph
        """
        # Fist graph convolution layer
        x1 = GraphConv(self.config.DIM_CONV1)([x, adj])
        x1 = KL.LeakyReLU()(x1)
        # Second graph convolution layer
        x2 = GraphConv(self.config.DIM_CONV2)([x1, adj])
        x2 = KL.LeakyReLU()(x2)
        # Residual connection
        x2 = KL.Add()([x2, x1])
        # Third graph convolution layer
        x3 = GraphConv(self.config.DIM_CONV3)([x2, adj])
        x3 = KL.LeakyReLU()(x3)
        # Residual connection
        x3 = KL.Add()([x3, x2])

        h = ReadoutLayer(self.config.DIM_READOUT)(x3)
        h = KL.LeakyReLU()(h)

        return h
