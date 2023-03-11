#!/usr/bin/env python
# coding: utf-8
from kerastuner import HyperModel
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tqdm.autonotebook import tqdm
import numpy as np
from set_seed import *
from read_calculate_dim_x import *
from vae_cost_log_loss import *


# added the following class implementation later for clarification;
# code is based on https://www.tensorflow.org/tutorials/generative/cvae#network_architecture
class EncoderZ(tfkl.Layer):

    def __init__(self, seq_len, epochs, min_delta, seed_value,
                 encoder_dim_two_num, encoder_dim_three_num, encoder_dim_four_num,
                 encoder_dim_one_activation, encoder_dim_two_activation,
                 encoder_dim_three_activation, encoder_dim_z,
                 name="encoder", **kwargs):
        super(EncoderZ, self).__init__(name=name, **kwargs)
        SetSeed(seed_value)
        tf.random.set_seed(seed_value)
        self.dim_x = ReadXtrainCalculateDimX(data_dir='/users/k1754828/DATA/')
        self.seq_len = seq_len
        self.epochs = epochs
        self.min_delta = min_delta
        self.seed_value = seed_value
        self.encoder_dim_two_num = encoder_dim_two_num
        self.encoder_dim_three_num = encoder_dim_three_num
        self.encoder_dim_four_num = encoder_dim_four_num
        self.encoder_dim_one_activation = encoder_dim_one_activation
        self.encoder_dim_two_activation = encoder_dim_two_activation
        self.encoder_dim_three_activation = encoder_dim_three_activation
        self.encoder_dim_one_activation = encoder_dim_one_activation
        self.encoder_dim_two_activation = encoder_dim_two_activation
        self.encoder_dim_three_activation = encoder_dim_three_activation
        self.encoder_dim_z = encoder_dim_z

    def gru_layer_one(self):
        return tfkl.Bidirectional(tfkl.GRU(self.dim_x, activation=self.encoder_dim_one_activation,
                                           kernel_initializer="glorot_uniform",
                                           bias_initializer="zeros",
                                           dropout=self.encoder_dim_dropout_one_num, return_sequences=True))

    def gru_layer_two(self):
        return tfkl.Bidirectional(tfkl.GRU(self.encoder_dim_two_num, activation=self.encoder_dim_two_activation,
                                           kernel_initializer="glorot_uniform",
                                           bias_initializer="zeros",
                                           dropout=self.encoder_dim_dropout_two_num, return_sequences=True))

    def gru_layer_three(self):
        return tfkl.Bidirectional(
            tfkl.GRU(self.encoder_dim_dropout_three_num, activation=self.encoder_dim_activation_three_num,
                     kernel_initializer="glorot_uniform",
                     bias_initializer="zeros",
                     dropout=self.encoder_dim_dropout_three_num, return_sequences=True))

    def gru_layer_four(self):
        return tfkl.Bidirectional(tfkl.GRU(self.encoder_dim_four_num, activation=self.encoder_dim_activation_four_num,
                                           kernel_initializer="glorot_uniform",
                                           bias_initializer="zeros",
                                           dropout=self.encoder_dim_dropout_four_num, return_sequences=True))

    def build(self):
        layers = [tfkl.InputLayer(input_shape=(self.seq_len, self.dim_x)),
                  self.gru_layer_one(),
                  tfkl.BatchNormalization(axis=1, momentum=0.1, epsilon=0.001),
                  self.gru_layer_two(),
                  tfkl.BatchNormalization(axis=1, momentum=0.1, epsilon=0.001),
                  self.gru_layer_three(),
                  tfkl.BatchNormalization(axis=1, momentum=0.1, epsilon=0.001),
                  self.gru_layer_four(),
                  tfkl.BatchNormalization(axis=1, momentum=0.1, epsilon=0.001),
                  tfkl.Flatten(), tfkl.Dense(self.encoder_dim_z * 2, activation=None),
                  tfkl.BatchNormalization(axis=1, momentum=0.1, epsilon=0.001)]
        return tfk.Sequential(layers)

    def get_config(self):
        config = super(EncoderZ, self).get_config()
        config.update({
            'dim_x': self.dim_x,
            'seq_len': self.seq_len,
            'epochs': self.epochs,
            'seed_value': self.seed_value,
            'encoder_dim_two_num': self.encoder_dim_two_num,
            'encoder_dim_three_num': self.encoder_dim_three_num,
            'encoder_dim_four_num': self.encoder_dim_two_num,
            'encoder_dim_one_activation': self.encoder_dim_one_activation,
            'encoder_dim_two_activation': self.encoder_dim_two_activation,
            'encoder_dim_three_activation': self.encoder_dim_three_activation,
            'encoder_dim_four_activation': self.encoder_dim_four_activation,
            'encoder_dim_z': self.encoder_dim_z,
        })
        return {"a": self.var.numpy()}


class DecoderX(tfk.layers.Layer):

    def __init__(self, seed_value, dim_z,
                 decoder_dim_two_num, decoder_dim_three_num, decoder_dim_four_num,
                 decoder_dim_one_activation, decoder_dim_two_activation, decoder_dim_three_activation,
                 decoder_dense_activation, name="decoder", **kwargs):
        super(DecoderX, self).__init__(name=name, **kwargs)
        self.seed_value = seed_value
        self.dim_z = dim_z
        self.decoder_dim_two_num = decoder_dim_two_num
        self.decoder_dim_three_num = decoder_dim_three_num
        self.decoder_dim_four_num = decoder_dim_four_num
        self.decoder_dim_one_activation = decoder_dim_one_activation
        self.decoder_dim_two_activation = decoder_dim_two_activation
        self.decoder_dim_three_activation = decoder_dim_three_activation
        self.decoder_dense_activation = decoder_dense_activation
        self.dim_x = ReadXtrainCalculateDimX(data_dir='/users/k1754828/DATA')
        SetSeed(self.seed_value)
        tf.random.set_seed(seed_value)

    def gru_layer_one(self):
        return tfkl.Bidirectional(tfkl.GRU(self.dim_z, activation=self.decoder_dim_one_activation,
                                           kernel_initializer="glorot_uniform",
                                           bias_initializer="zeros",
                                           dropout=self.decoder_dim_dropout_one_num, return_sequences=True))

    def gru_layer_two(self):
        return tfkl.Bidirectional(tfkl.GRU(self.decoder_dim_two_num, activation=self.decoder_dim_two_activation,
                                           kernel_initializer="glorot_uniform",
                                           bias_initializer="zeros",
                                           dropout=self.decoder_dim_dropout_two_num, return_sequences=True))

    def gru_layer_three(self):
        return tfkl.Bidirectional(
            tfkl.GRU(self.decoder_dim_dropout_three_num, activation=self.decoder_dim_activation_two_num,
                     kernel_initializer="glorot_uniform",
                     bias_initializer="zeros",
                     dropout=self.decoder_dim_dropout_two_num, return_sequences=True))

    def gru_layer_four(self):
        return tfkl.Bidirectional(tfkl.GRU(self.dim_x, activation=self.decoder_dim_four_activations,
                                           kernel_initializer="glorot_uniform",
                                           bias_initializer="zeros",
                                           dropout=self.decoder_dim_four_dropout, return_sequences=True))

    def build(self):
        layers = [tfkl.InputLayer(input_shape=(self.seq_len, self.dim_z,)),
                  self.gru_layer_one(),
                  tfkl.BatchNormalization(axis=1, momentum=0.1, epsilon=0.001),
                  self.gru_layer_two(),
                  tfkl.BatchNormalization(axis=1, momentum=0.1, epsilon=0.001),
                  self.gru_layer_three(),
                  tfkl.BatchNormalization(axis=1, momentum=0.1, epsilon=0.001),
                  self.gru_layer_four(),
                  tfkl.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0011),
                  ]

        return tfk.Sequential(layers)

    def get_config(self):
        config = super(DecoderX, self).get_config()
        config.update({
            'data': self.data,
            'epochs': self.epochs,
            'min_delta': self.min_delta,
            'batch_size': self.batch_size,
            'seed_value': self.seed_value,
            'set_seed': self.set_seed,
        })
        return {"a": self.var.numpy()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SamplerZ(tfk.layers.Layer):
    def __init__(self, dim_z, seed_value, *args, **kwargs):
        super(SamplerZ, self).__init__(*args, **kwargs)
        self.seed_value = seed_value
        self.dim_z = dim_z

    def call(self, inputs):
        mu, rho = inputs
        tf.random.set_seed(self.seed_value)
        sd = tf.math.log(1 + tf.math.exp(rho))
        z_sample = mu + sd * tf.random.normal(seed=self.seed_value, shape=(self.dim_z,))
        return z_sample

    def get_config(self):
        config = super(SamplerZ, self).get_config()
        config.update({
            'seed_value': self.seed_value,
            'dim_z': self.dim_z,
        })
        return {"a": self.var.numpy()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ConstructModel():
    def __init__(self, dim_z, encoder_dim_two_num, encoder_dim_three_num, encoder_dim_four_num,
                 encoder_dim_one_activation, encoder_dim_two_activation, encoder_dim_three_activation,
                 encoder_dim_four_activation,
                 encoder_dim_one_dropout, encoder_dim_two_dropout, encoder_dim_three_dropout,
                 encoder_dim_four_dropout,
                 decoder_dim_one_num, decoder_dim_two_num, decoder_dim_three_num, decoder_dim_four_num,
                 decoder_dim_one_activation, decoder_dim_two_activation, decoder_dim_three_activation,
                 decoder_dim_four_activation,
                 decoder_dim_one_dropout, decoder_dim_two_dropout, decoder_dim_three_dropout,
                 decoder_dim_four_dropout,
                 seed_value, *args, **kwargs):
        super(ConstructModel, self).__init__(*args, **kwargs)

        self.dim_z = dim_z
        self.encoder_dim_two_num = encoder_dim_two_num
        self.encoder_dim_three_num = encoder_dim_three_num
        self.encoder_dim_four_num = encoder_dim_four_num
        self.encoder_dim_one_activation = encoder_dim_one_activation
        self.encoder_dim_two_activation = encoder_dim_two_activation
        self.encoder_dim_three_activation = encoder_dim_three_activation
        self.encoder_dim_four_activation = encoder_dim_four_activation
        self.encoder_dim_one_dropout = encoder_dim_one_dropout
        self.encoder_dim_two_dropout = encoder_dim_two_dropout
        self.encoder_dim_three_dropout = encoder_dim_three_dropout
        self.encoder_dim_four_dropout = encoder_dim_four_dropout
        self.decoder_dim_one_num = decoder_dim_one_num
        self.decoder_dim_two_num = decoder_dim_two_num
        self.decoder_dim_three_num = decoder_dim_three_num
        self.decoder_dim_four_num = decoder_dim_four_num
        self.decoder_dim_one_activation = decoder_dim_one_activation
        self.decoder_dim_two_activation = decoder_dim_two_activation
        self.decoder_dim_three_activation = decoder_dim_three_activation
        self.decoder_dim_four_activation = decoder_dim_four_activation
        self.decoder_dim_one_dropout = decoder_dim_one_dropout
        self.decoder_dim_two_dropout = decoder_dim_two_dropout
        self.decoder_dim_three_dropout = decoder_dim_three_dropout
        self.decoder_dim_four_dropout = decoder_dim_four_dropout
        self.seed_value = seed_value
        self.dim_x = ReadXtrainCalculateDimX(data_dir='/users/k1754828/DATA/')

        self.encoder = EncoderZ(seed_value=self.seed_value,
                                dim_z=dim_z,
                                encoder_dim_two_num=self.encoder_dim_two_num,
                                encoder_dim_three_num=self.encoder_dim_three_num,
                                encoder_dim_four_num=self.encoder_dim_four_num,
                                encoder_dim_one_activation=self.encoder_dim_one_activation,
                                encoder_dim_two_activation=self.encoder_dim_two_activation,
                                encoder_dim_three_activation=self.encoder_dim_three_activation,
                                encoder_dim_four_activation=self.encoder_dim_four_activation,
                                encoder_dim_one_dropout=self.encoder_dim_one_dropout,
                                encoder_dim_two_dropout=self.encoder_dim_two_dropout,
                                encoder_dim_three_dropout=self.encoder_dim_three_dropout,
                                encoder_dim_four_dropout=self.encoder_dim_four_dropout,
                                dim_x=self.dim_x).build()
        self.decoder = DecoderX(seed_value=self.seed_value,
                                dim_z=self.dim_z, decoder_dim_one_num=self.decoder_dim_one_num,
                                decoder_dim_two_num=self.decoder_dim_two_num,
                                decoder_dim_three_num=self.decoder_dim_three_num,
                                decoder_dim_four_num=self.decoder_dim_four_num,
                                decoder_dim_one_activation=self.decoder_dim_one_activation,
                                decoder_dim_two_activation=self.decoder_dim_two_activation,
                                decoder_dim_three_activation=self.decoder_dim_three_activation,
                                decoder_dim_four_activation=self.decoder_dim_four_activation,
                                decoder_dim_one_dropout=self.decoder_dim_one_dropout,
                                decoder_dim_two_dropout=self.decoder_dim_two_dropout,
                                decoder_dim_three_dropout=self.decoder_dim_three_dropout,
                                decoder_dim_four_dropout=self.decoder_dim_four_dropout,
                                dim_x=self.dim_x).build()
        self.sampler = SamplerZ(dim_z=self.dim_z, seed_value=self.seed_value)

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'seed_value': self.seed_value,
            'dim_z': self.dim_z,
            'encoder_dim_two_num': self.encoder_dim_two_num,
            'encoder_dim_three_num': self.encoder_dim_three_num,
            'encoder_dim_four_num': self.encoder_dim_four_num,
            'encoder_dim_one_activation': self.encoder_dim_one_activation,
            'encoder_dim_two_activation': self.encoder_dim_two_activation,
            'encoder_dim_three_activation': self.encoder_dim_three_activation,
            'encoder_dim_four_activation': self.encoder_dim_four_activation,
            'encoder_dim_one_dropout': self.encoder_dim_one_dropout,
            'encoder_dim_two_dropout': self.encoder_dim_two_dropout,
            'encoder_dim_three_dropout': self.encoder_dim_three_dropout,
            'encoder_dim_four_dropout': self.encoder_dim_four_dropout,
            'decoder_dim_one_num': self.decoder_dim_one_num,
            'decoder_dim_two_num': self.decoder_dim_two_num,
            'decoder_dim_three_num': self.decoder_dim_three_num,
            'decoder_dim_one_activation': self.decoder_dim_one_activation,
            'decoder_dim_two_activation': self.decoder_dim_two_activation,
            'decoder_dim_three_activation': self.decoder_dim_three_activation,
            'decoder_dim_four_activation': self.decoder_dim_four_activation,
            'decoder_dim_one_dropout': self.decoder_dim_one_dropout,
            'decoder_dim_two_dropout': self.decoder_dim_two_dropout,
            'decoder_dim_three_dropout': self.decoder_dim_three_dropout,
            'decoder_dim_four_dropout': self.decoder_dim_four_dropout,
            'dim_x': self.dim_x,
        })
        return {"a": self.var.numpy()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class VAE(HyperModel):
    def __init__(self, data, epochs, min_delta, batch_size, seed_value,
                 name="autoencoder", **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.dim_z = None

        self.encoder_dim_one_num = None
        self.encoder_dim_one_activation = None
        self.encoder_dim_one_dropout = None

        self.encoder_dim_two_num = None
        self.encoder_dim_two_activation = None
        self.encoder_dim_two_dropout = None

        self.encoder_dim_three_num = None
        self.encoder_dim_three_activation = None
        self.encoder_dim_three_dropout = None

        self.encoder_dim_four_num = None
        self.encoder_dim_four_activation = None
        self.encoder_dim_four_dropout = None

        self.decoder_dim_one_num = None
        self.decoder_dim_one_activation = None
        self.decoder_dim_one_dropout = None

        self.decoder_dim_two_num = None
        self.decoder_dim_two_activation = None
        self.decoder_dim_two_dropout = None

        self.decoder_dim_three_num = None
        self.decoder_dim_three_activation = None
        self.decoder_dim_three_dropout = None

        self.decoder_dim_four_num = None
        self.decoder_dim_four_activation = None
        self.decoder_dim_four_dropout = None

        self.dim_x = None

        self.model = None

        self.data = data

        self.epochs = epochs

        self.min_delta = min_delta

        self.batch_size = batch_size

        self.seed_value = seed_value

        self.set_seed = SetSeed(self.seed_value)

        tf.random.set_seed(self.seed_value)

    def encode(self, x_input):
        mu, sd = tf.split(self.encoder(x_input), num_or_size_splits=2, axis=1)
        z_sample = self.sampler([mu, sd])
        return z_sample, mu, sd

    def vae_cost(self, x_true, model, analytic_kl=True, kl_weight=1):
        z_sample, mu, sd = model.encode(x_true)
        x_recons_logits = model.decoder(z_sample)
        x_true = np.asarray(x_true).astype('float32')

        x_true = tf.convert_to_tensor(x_true)
        neg_log_likelihood = custom_sigmoid_cross_entropy_loss_with_logits(x_true, x_recons_logits)

        if analytic_kl:
            kl_divergence = - 0.5 * tf.math.reduce_sum(
                1 + tf.math.log(tf.math.square(sd)) - tf.math.square(mu) - tf.math.square(sd),
                axis=1)
        else:
            logpz = normal_log_pdf(z_sample, 0., 1.)
            logqz_x = normal_log_pdf(z_sample, mu, tf.math.square(sd))
            kl_divergence = logqz_x - logpz
        elbo = tf.math.reduce_mean(-kl_weight * kl_divergence - neg_log_likelihood)
        return -elbo

    def train_step(self, x_true, model, analytic_kl=True, kl_weight=1):
        with tf.GradientTape() as tape:
            cost_mini_batch = self.vae_cost(x_true, model, analytic_kl, kl_weight)

            print('\n' + str(np.asarray(cost_mini_batch)))
        gradients = tape.gradient(cost_mini_batch, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return self

    def build(self, hp):

        self.dim_z = hp.Int('dim_z', min_value=32, max_value=256, step=32)

        self.encoder_dim_one_num = hp.Int('encoder_dim_one_num', min_value=32, max_value=256, step=32)
        self.encoder_dim_one_activation = hp.Choice('encoder_dim_one_activation', ['relu', 'tanh'])
        self.encoder_dim_one_dropout = hp.Float('encoder_dim_one_dropout', min_value=0.01, max_value=1, step=0.01)

        self.encoder_dim_two_num = hp.Int('encoder_dim_two_num', min_value=32, max_value=256, step=32)
        self.encoder_dim_two_activation = hp.Choice('encoder_dim_two_activation', ['relu', 'tanh'])
        self.encoder_dim_two_dropout = hp.Float('encoder_dim_two_dropout', min_value=0.01, max_value=1, step=0.01)

        self.encoder_dim_three_num = hp.Int('encoder_dim_three_num', min_value=32, max_value=256, step=32)
        self.encoder_dim_three_activation = hp.Choice('encoder_dim_three_activation', ['relu', 'tanh'])
        self.encoder_dim_three_dropout = hp.Float('encoder_dim_three_dropout', min_value=0.01, max_value=1, step=0.01)

        self.encoder_dim_four_num = hp.Int('encoder_dim_four_num', min_value=32, max_value=256, step=32)
        self.encoder_dim_four_activation = hp.Choice('encoder_dim_four_activation', ['relu', 'tanh'])
        self.encoder_dim_four_dropout = hp.Float('encoder_dim_four_dropout', min_value=0.01, max_value=1, step=0.01)

        self.decoder_dim_one_num = hp.Int('decoder_dim_one_num', min_value=32, max_value=256, step=32)
        self.decoder_dim_one_activation = hp.Choice('decoder_dim_one_activation', ['relu', 'tanh'])
        self.decoder_dim_one_dropout = hp.Float('decoder_dim_one_dropout', min_value=0.01, max_value=1, step=0.01)

        self.decoder_dim_two_num = hp.Int('decoder_dim_two_num', min_value=32, max_value=256, step=32)
        self.decoder_dim_two_activation = hp.Choice('decoder_dim_two_activation', ['relu', 'tanh'])
        self.decoder_dim_two_dropout = hp.Float('decoder_dim_one_dropout', min_value=0.01, max_value=1, step=0.01)

        self.decoder_dim_three_num = hp.Int('decoder_dim_three_num', min_value=32, max_value=256, step=32)
        self.decoder_dim_three_activation = hp.Choice('decoder_dim_three_activation', ['relu', 'tanh'])
        self.decoder_dim_three_dropout = hp.Float('decoder_dim_three_dropout', min_value=0.01, max_value=1, step=0.01)

        self.decoder_dim_four_num = hp.Int('decoder_dim_four_num', min_value=32, max_value=256, step=32)
        self.decoder_dim_four_activation = hp.Choice('decoder_dim_four_activation', ['relu', 'tanh'])
        self.decoder_dim_four_dropout = hp.Float('decoder_dim_four_dropout', min_value=0.01, max_value=1, step=0.01)

        self.dim_x = ReadXtrainCalculateDimX(data_dir='/users/k1754828/DATA')

        # Construct Model
        self.model = ConstructModel(seed_value=self.seed_value,
                                    dim_z=self.dim_z,
                                    encoder_dim_one_num=self.encoder_dim_one_num,
                                    encoder_dim_two_num=self.encoder_dim_two_num,
                                    encoder_dim_three_num=self.encoder_dim_three_num,
                                    encoder_dim_one_activation=self.encoder_dim_one_activation,
                                    encoder_dim_two_activation=self.encoder_dim_two_activation,
                                    encoder_dim_three_activation=self.encoder_dim_three_activation,
                                    encoder_dim_one_dropout=self.encoder_dim_one_dropout,
                                    encoder_dim_two_dropout=self.encoder_dim_two_dropout,
                                    encoder_dim_three_dropout=self.encoder_dim_three_dropout,
                                    decoder_dim_one_num=self.decoder_dim_one_num,
                                    decoder_dim_two_num=self.decoder_dim_two_num,
                                    decoder_dim_three_num=self.decoder_dim_three_num,
                                    decoder_dim_four_num=self.decoder_dim_four_num,
                                    decoder_dim_one_activation=self.decoder_dim_one_activation,
                                    decoder_dim_two_activation=self.decoder_dim_two_activation,
                                    decoder_dim_three_activation=self.decoder_dim_three_activation,
                                    decoder_dim_four_activation=self.decoder_dim_four_activation,
                                    decoder_dim_one_dropout=self.decoder_dim_one_dropout,
                                    decoder_dim_two_dropout=self.decoder_dim_two_dropout,
                                    decoder_dim_three_dropout=self.decoder_dim_three_dropout,
                                    decoder_dim_four_dropout=self.decoder_dim_four_dropout,
                                    dim_x=self.dim_x)
        for epoch in range(self.epochs):
            for x in tqdm(self.data.xfull):
                x = tf.reshape(x, [self.data.dim_x, -1])
                x = tf.expand_dims(x, 0)
                x = tf.reshape(x, [-1, 1, self.data.dim_x])
                cost_mini_batch = self.train_step(x, self.model, analytic_kl=True, kl_weight=1)
                if cost_mini_batch == np.nan:
                    break;
            if epoch == 0:
                cost_mini_batch_epochwise = np.array(tf.reshape(cost_mini_batch, []))
            if epoch > 0:
                cost_mini_batch_epochwise_prev = cost_mini_batch_epochwise
                cost_mini_batch_epochwise = np.array(tf.reshape(cost_mini_batch, []))
                cost_mini_batch_delta = abs(cost_mini_batch_epochwise - cost_mini_batch_epochwise_prev)
                if cost_mini_batch < 2:
                    if cost_mini_batch_delta < self.min_delta:
                        return self.model

        return self.model

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'seed_value': self.seed_value,
            'set_seed': self.set_seed,
            'dim_z': self.dim_z,
            'encoder_dim_two_num': self.encoder_dim_two_num,
            'encoder_dim_three_num': self.encoder_dim_three_num,
            'encoder_dim_four_num': self.encoder_dim_four_num,
            'encoder_dim_one_activation': self.encoder_dim_one_activation,
            'encoder_dim_two_activation': self.encoder_dim_two_activation,
            'encoder_dim_three_activation': self.encoder_dim_three_activation,
            'encoder_dim_four_activation': self.encoder_dim_four_activation,
            'encoder_dim_one_dropout': self.encoder_dim_one_dropout,
            'encoder_dim_two_dropout': self.encoder_dim_two_dropout,
            'encoder_dim_three_dropout': self.encoder_dim_three_dropout,
            'encoder_dim_four_dropout': self.encoder_dim_four_dropout,
            'decoder_dim_one_num': self.decoder_dim_one_num,
            'decoder_dim_two_num': self.decoder_dim_two_num,
            'decoder_dim_three_num': self.decoder_dim_three_num,
            'decoder_dim_one_activation': self.decoder_dim_one_activation,
            'decoder_dim_two_activation': self.decoder_dim_two_activation,
            'decoder_dim_three_activation': self.decoder_dim_three_activation,
            'decoder_dim_four_activation': self.decoder_dim_four_activation,
            'decoder_dim_one_dropout': self.decoder_dim_one_dropout,
            'decoder_dim_two_dropout': self.decoder_dim_two_dropout,
            'decoder_dim_three_dropout': self.decoder_dim_three_dropout,
            'decoder_dim_four_dropout': self.decoder_dim_four_dropout,
            'dim_x': self.dim_x,
        })
        return {"a": self.var.numpy()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)