from tensorflow.keras import layers
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures

#https://keras.io/examples/keras_recipes/quasi_svm/

class random_fourier_features_layer(layers.Layer):
    def __init__(self, output_dim, scale, kernel_initializer, **kwargs):
        super(random_fourier_features_layer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.scale = scale
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.rff_layer = RandomFourierFeatures(
            output_dim=self.output_dim, scale=self.scale,
            kernel_initializer=self.kernel_initializer,
        )
        return self

    def get_config(self):
        cfg = super().get_config()
        return cfg


    def call(self, inputs):
        y = self.rff_layer(inputs)
        return y
