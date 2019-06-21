import keras
import tensorflow as tf
import keras.backend as K
import keras.layers as KL


class GraphLayer(KL.Layer):

    def __init__(self,
                 activation=None,
                 **kwargs):
        """Initialize the layer.
        :param activation: The activation function after convolution.
        :param kwargs: Other arguments for parent class.
        """
        self.supports_masking = True
        self.activation = keras.activations.get(activation)
        self.supports_masking = True
        super(GraphLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'activation': self.activation,
        }
        base_config = super(GraphLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        features, edges = inputs
        edges = K.cast(edges, K.floatx())
        outputs = self.activation(self._call(features, edges))
        return outputs

    def _call(self, features, edges):
        raise NotImplementedError('The class is not intended to be used directly.')


class GraphConv(GraphLayer):
    """Graph convolutional layer."""

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize the layer.
        :param units: Number of new states. If the input shape is (batch_size, node_num, feature_len), then the output
                      shape is (batch_size, node_num, units).
        :param kernel_initializer: The initializer of the kernel weight matrix.
        :param kernel_regularizer: The regularizer of the kernel weight matrix.
        :param kernel_constraint:  The constraint of the kernel weight matrix.
        :param use_bias: Whether to use bias term.
        :param bias_initializer: The initializer of the bias vector.
        :param bias_regularizer: The regularizer of the bias vector.
        :param bias_constraint: The constraint of the bias vector.
        :param kwargs: Other arguments for parent class.
        """
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.W, self.b = None, None
        self.W1, self.b1 = None, None
        super(GraphConv, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'use_bias': self.use_bias,
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        feature_dim = input_shape[0][2]
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        self.W1 = self.add_weight(
            shape=(feature_dim + 2 * self.units, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b1'.format(self.name),
            )
        super(GraphConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (self.units,)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def _call(self, features, edges):
        new_features = K.dot(features, self.W)
        if self.use_bias:
            new_features += self.b

        message_in = K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), new_features)
        message_out = K.batch_dot(edges, new_features)

        concat = K.concatenate([message_in, message_out, features])

        updated_message = K.dot(concat, self.W1)
        if self.use_bias:
            updated_message += self.b1

        return updated_message


class ReadoutLayer(KL.Layer):
    """Generates the graph latent representation from given node representations."""

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation=None,
                 **kwargs):
        """Initialize the layer.
        :param activation: The activation function.
        :param kwargs: Other arguments for parent class.
        """
        self.supports_masking = True
        self.activation = keras.activations.get(activation)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.W, self.b = None, None
        super(ReadoutLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'activation': self.activation,
            'units': self.units,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'use_bias': self.use_bias,
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(ReadoutLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        feature_dim = input_shape[2]
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
        super(ReadoutLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (self.units,)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, **kwargs):
        features = inputs
        outputs = self.activation(self._call(features))
        return outputs

    def _call(self, features):
        features = K.dot(features, self.W)
        if self.use_bias:
            features += self.b

        features = K.sum(features, axis=1)

        return features
