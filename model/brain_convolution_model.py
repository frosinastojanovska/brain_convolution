import tensorflow as tf
import keras.backend as K
import keras.models as KM
import keras.layers as KL


class StructureMessageInput(KL.Layer):
    """Structure the input for the message encoder"""

    def __init__(self, **kwargs):
        super(StructureMessageInput, self).__init__(**kwargs)

    def call(self, inputs):
        adj = inputs[0]
        x = inputs[1]
        adj_neighbors_feat = tf.map_fn(lambda xy: tf.multiply(xy[0], xy[1]),
                                       elems=(adj, x),
                                       dtype=tf.float32)
        concat_mess = tf.concat([adj, adj_neighbors_feat], axis=-1)
        return concat_mess

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], input_shape[1][1], input_shape[1][1], input_shape[1][2]


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
        """ Build Deep Protein Annotation architecture.
        :return: keras model
        :rtype: KM.Model
        """
        x = KL.Input(batch_shape=[self.config.BATCH_SIZE, None, self.config.NODE_DIM],
                     name='input_node_features', dtype=tf.float32)

        adj = KL.Input(batch_shape=[self.config.BATCH_SIZE, None, None, 1],
                       name='input_adjacency_matrix', dtype=tf.float32)
        encoded_state = self.graph_encoder(x, adj)

        dense_out = KL.Dense(128, name='dense_layer1')(encoded_state)
        dense_out = KL.LeakyReLU()(dense_out)
        dense_out = KL.Dropout(0.1)(dense_out)
        dense_out = KL.Dense(64, name='dense_layer2')(dense_out)
        dense_out = KL.LeakyReLU()(dense_out)
        dense_out = KL.Dropout(0.1)(dense_out)
        output = KL.Dense(self.config.NUM_CLASSES, activation='softmax', name='dense_layer3')(dense_out)

        return KM.Model([x, adj], output, name='brainconvolution')

    def graph_encoder(self, x, adj):
        """Encode graph into vector representation with graph convolution network

        :param x: hidden representation of nodes
        :param adj: adjacency matrix including the edge features
        :return: latent representation of graph
        """
        # Fist graph convolution layer
        x1 = self.graph_convolution(x, adj, self.config.NODE_DIM + 1,
                                    self.config.DIM_CONV1, '1')
        # Second graph convolution layer
        x2 = self.graph_convolution(x1, adj, self.config.DIM_CONV1 + 1,
                                    self.config.DIM_CONV2, '2')
        # Residual connection
        x2 = KL.Add()([x2, x1])
        # Third graph convolution layer
        x3 = self.graph_convolution(x2, adj, self.config.DIM_CONV2 + 1,
                                    self.config.DIM_CONV3, '3')
        # Residual connection
        x3 = KL.Add()([x3, x2])

        h = self.readout_function(x3, self.config.DIM_READOUT)

        return h

    def graph_convolution(self, x, adj, feature_dim, dim, suffix):
        """Logic for graph convolution layer

        :param x: hidden representation of nodes
        :param adj: adjacency matrix including the edge features
        :param feature_dim: dimension of the input features
        :param dim: dimension of the hidden representations
        :param suffix: suffix for the layer names
        :return: updated node hidden representations
        """
        message_model = self.message_model(dim, feature_dim, suffix)
        summation = KL.Lambda(lambda k: K.sum(k, axis=-2), name='message_sum' + suffix)
        # message from edges
        message_features = StructureMessageInput()([adj, x])
        message_features = KL.Masking()(message_features)
        mess = KL.TimeDistributed(KL.TimeDistributed(message_model))(message_features)
        mess = summation(mess)
        # update node representation vectors
        h = self.update_function(x, mess, dim, suffix)

        return h

    def message_model(self, dim, feature_dim, suffix='1'):
        """Creates message from the neighborhood

        :param dim: dimension of the output message
        :param feature_dim: dimension of the input features
        :param suffix: suffix for the layer names
        :return: generated messages
        """
        x = KL.Input(batch_shape=[self.config.BATCH_SIZE, None, feature_dim],
                     name='input_features' + suffix)

        net = KL.Dense(dim, name='message_FC' + suffix)(x)
        net = KL.LeakyReLU()(net)
        mess = KL.Dropout(0.1)(net)

        return KM.Model(x, mess, name='message_model' + suffix)

    @staticmethod
    def update_function(x, mess, dim, suffix):
        """ Update function for new node representations

        :param x: input, node representations
        :param mess: message from edges
        :param dim: dimension of the output
        :param suffix: suffix for the layer names
        :return: updated node representations
        """
        d = KL.Concatenate(name='message_concat' + suffix)([x, mess])
        net = KL.TimeDistributed(KL.Dense(dim, name='update_FC' + suffix))(d)
        net = KL.TimeDistributed(KL.LeakyReLU())(net)
        out = KL.Dropout(0.1)(net)

        return out

    @staticmethod
    def readout_function(x, dim):
        """Generates the graph latent representation from given node representations

        :param x: node representations
        :param dim: dimension of the output
        :return: graph representation
        """
        net = KL.TimeDistributed(KL.Dense(dim, use_bias=False, name='readout_FC'))(x)
        net = KL.TimeDistributed(KL.LeakyReLU())(net)
        net = KL.Dropout(0.1)(net)

        out = KL.Lambda(lambda k: K.sum(k, axis=-2), name='readout_sum')(net)
        return out
