from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import InputSpec
from keras.engine.base_layer import Layer
from keras.legacy import interfaces


class STUnit(Layer):
    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(STUnit, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # input_shape: (None, 16/256/128)
        # input_dim: 16/256/128
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units * 2),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 2,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_g = self.kernel[:, self.units:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_g = self.bias[self.units:]
        else:
            self.bias_i = None
            self.bias_g = None

        self.built = True

    def call(self, inputs):
        inputs_i = inputs
        inputs_g = inputs

        x_i = K.dot(inputs_i, self.kernel_i)
        x_g = K.dot(inputs_g, self.kernel_g)

        if self.use_bias:
            x_i = K.bias_add(x_i, self.bias_i)
            x_g = K.bias_add(x_g, self.bias_g)

        i = self.activation(x_i)  # tanh
        g = self.recurrent_activation(x_g)  # hard_sigmoid
        output = i * g
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(STUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GatedDense(Layer):
    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 recurrent_activation='hard_sigmoid',
                 # recurrent_activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GatedDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # input_shape: (None, 16/256/128)
        # input_dim: 16/256/128
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_c = self.kernel[:, self.units: self.units * 2]
        self.kernel_o = self.kernel[:, self.units * 2:]
        # self.kernel_f = self.kernel[:, self.units: self.units * 2]
        # self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        # self.kernel_o = self.kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_c = self.bias[self.units: self.units * 2]
            self.bias_o = self.bias[self.units * 2:]
            # self.bias_f = self.bias[self.units: self.units * 2]
            # self.bias_c = self.bias[self.units * 2: self.units * 3]
            # self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            # self.bias_f = None
            self.bias_c = None
            self.bias_o = None

        self.built = True

    def call(self, inputs):
        inputs_i = inputs
        # inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs

        x_i = K.dot(inputs_i, self.kernel_i)
        # x_f = K.dot(inputs_f, self.kernel_f)
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = K.dot(inputs_o, self.kernel_o)

        if self.use_bias:
            x_i = K.bias_add(x_i, self.bias_i)
            # x_f = K.bias_add(x_f, self.bias_f)
            x_c = K.bias_add(x_c, self.bias_c)
            x_o = K.bias_add(x_o, self.bias_o)
            # print('x_i:',x_i.shape)

        i = self.recurrent_activation(x_i)
        # f = self.recurrent_activation(x_f)
        c = i * self.activation(x_c)
        o = self.recurrent_activation(x_o)
        output = o * self.activation(c)
        # output = K.dot(inputs, self.kernel)
        # if self.use_bias:
        #     output = K.bias_add(output, self.bias)
        # if self.activation is not None:
        #     output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(GatedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MinGatedDense(Layer):

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MinGatedDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # input_shape: (None, 16/256/128)
        # input_dim: 16/256/128
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units * 2),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 2,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_c = self.kernel[:, self.units:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_c = self.bias[self.units:]
        else:
            self.bias_i = None
            self.bias_c = None

        self.built = True

    def call(self, inputs):
        inputs_i = inputs
        inputs_c = inputs

        x_i = K.dot(inputs_i, self.kernel_i)
        x_c = K.dot(inputs_c, self.kernel_c)

        if self.use_bias:
            x_i = K.bias_add(x_i, self.bias_i)
            x_c = K.bias_add(x_c, self.bias_c)

        i = self.recurrent_activation(x_i)
        c = i * self.activation(x_c)
        output = self.activation(c)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MinGatedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




