# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from keras.datasets import mnist

from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Flatten, Activation, Reshape, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


class mnist_GAN():
    def __init__(self):
        self.width = 28
        self.height = 28
        self.channels = 1
        # 784 
        self.img_shape = (self.width, self.height, self.channels)

        self.noise_dim = 100

        #! Adam的具体参数设置对结果有什么影响？
        self.OPTIMIZER = optimizers.Adam(lr=0.0002,beta_1=0.5)

        self.generator = self.generator()

        self.discriminator = self.discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])

        #!discriminator参数的固定不应该在这里？
        # self.discriminator.trainable = False

        self.stacked = self.stacked_G_D()
        self.stacked.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

    def generator(self):
        """Generate image from noise.
        """
        model = Sequential()

        model.add(Dense(256, input_shape=(self.noise_dim,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.width * self.height * self.channels))
        model.add(Activation(activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        return model

    def discriminator(self):
        """Discriminate whether a avatar is real.
        """
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.02))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))
        model.add(Activation(activation='sigmoid'))

        model.summary()

        return model

    def stacked_G_D(self):
        self.discriminator.trainable = False
        model = Sequential()

        model.add(self.generator)
        model.add(self.discriminator)

        return model

    def train(self, train_data, epochs=30000, batch_size=64, save_interval=500):
        """Train the generative adversarial model to generate anime avatar.
        """
        for epoch in range(epochs):
            #*训练discriminator
            random_idx = np.random.randint(0, train_data.shape[0], batch_size)
            real_img = train_data[random_idx]
            real_label = np.ones((batch_size,1))

            # random.normal()返回的直接是指定的(batch_size, noise_dim)的形状
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            fake_img = self.generator.predict(noise)
            fake_label = np.zeros((batch_size,1))

            d_loss_real = self.discriminator.train_on_batch(real_img, real_label)
            d_loss_fake = self.discriminator.train_on_batch(fake_img, fake_label)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #*训练generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            confusion_label = np.ones((batch_size, 1))

            g_loss = self.stacked.train_on_batch(noise, confusion_label)

            #*输出训练信息
            print("Epoch: %5d, [D loss: %f] [D acc: %3.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_sample(epoch)

    def save_sample(self, epoch):
        """Sample image in given interval.
        """
        # canvas size
        row, column = 5,5
        # 生成展示图片
        noise = np.random.normal(0, 1, (row * column, self.noise_dim))
        gen_img = self.generator.predict(noise)

        # 将像素值调整到 0-1
        gen_img = 0.5 * gen_img + 0.5

        fig, axs = plt.subplots(row, column)
        cnt = 0
        for r in range(row):
            for c in range(column):
                axs[r,c].imshow(gen_img[cnt,:,:,0], cmap='gray')
                axs[r,c].axis('off')
                cnt += 1
        fig.savefig('/home/xingyun/gan/gan_mnist_result/%d.png' % epoch)
        plt.close()

if __name__ == '__main__':
    (mnist_data,_),(_,_) = mnist.load_data('/data0/xingyun/MNIST/mnist.npz')
    mnist_data = mnist_data/127.5 - 1
    mnist_data = np.expand_dims(mnist_data, axis=3)
    
    mnist_gan = mnist_GAN()
    mnist_gan.train(train_data=mnist_data, epochs=30000, batch_size=32, save_interval=200)