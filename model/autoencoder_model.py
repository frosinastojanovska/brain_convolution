import tensorflow as tf
import keras.models as KM
import keras.layers as KL

from model.layers import GraphConv, Decoder


class BrainAutoencoderModel:
    """
    BrainAutoencoder model.
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

        decoded_state = Decoder(name='decoder')(encoded_state)

        return KM.Model([x, adj], decoded_state, name='autoencoder')

    def graph_encoder(self, x, adj):
        """Encode graph into vector representation with graph convolution network

        :param x: hidden representation of nodes
        :param adj: adjacency matrix including the edge features
        :return: latent representation of graph
        """
        # Fist graph convolution layer
        x1 = GraphConv(self.config.DIM_CONV1)([x, adj])
        x1 = KL.BatchNormalization(axis=-1)(x1)
        x1 = KL.LeakyReLU()(x1)
        x1 = KL.Dropout(0.1)(x1)
        # Second graph convolution layer
        x2 = GraphConv(self.config.DIM_CONV2)([x1, adj])
        x2 = KL.BatchNormalization(axis=-1)(x2)
        x2 = KL.LeakyReLU()(x2)
        x2 = KL.Dropout(0.1)(x2)
        # Residual connection
        x2 = KL.Add()([x2, x1])
        # Third graph convolution layer
        x3 = GraphConv(self.config.DIM_CONV3)([x2, adj])
        x3 = KL.BatchNormalization(axis=-1)(x3)
        x3 = KL.LeakyReLU()(x3)
        x3 = KL.Dropout(0.1)(x3)
        # Residual connection
        x3 = KL.Add()([x3, x2])

        return x3

    def build_encoder(self):
        x = KL.Input(shape=[None, self.config.NODE_DIM],
                     name='input_node_features', dtype=tf.float32)

        adj = KL.Input(shape=[None, None],
                       name='input_adjacency_matrix', dtype=tf.float32)

        encoded_state = self.graph_encoder(x, adj)
        return KM.Model([x, adj], encoded_state, name='encoder')
