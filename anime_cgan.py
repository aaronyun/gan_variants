# -*- coding:utf-8 -*-
import numpy as np

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization, Activation, Reshape, Flatten, Input

class CGAN():
    """
    """
    def __init__(self):
        self.width = 28
        self.height = 28
        self.channels = 1
        self.img_shape = (self.width, self.height, self.channels)

        self.cls_nums = 10
        self.noise_dim = 100

        self.OPTIMIZER = optimizers.Adam(lr=0.0002)


    def build_generator(self):
        """
        """
        model = Sequential()

        model.add(Dense(256, input_shape=(self.noise_dim,)))
        model.add(LeakyReLU(alphs=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alphs=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alphs=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape)))
        model.add(Activation(activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        # 模型输入
        noise = Input(shape=(self.noise_dim,))
        label = Input(shape=(1,), dtype='int32')
        model_input = 

        # 模型输出
        img = model(model_input)

        return Model(inputs=[noise, label], outputs=img)

    def discriminator(self):
        """
        """
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.noise_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))
        model.add(Activation(activation='sigmoid'))

        model.summary()

        return model

    def stacked_g_d(self):
        """
        """

    def train(self, train_data, epochs, batch_size=32, save_interval=200):
        """
        """

if __name__ == '__main__':
    cgan = CGAN()
    train_data = 
    cgan.train(train_data, epochs=30000, batch_size=32, save_interval=200)