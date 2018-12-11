# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import image

from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Reshape, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


class anime_GAN():
    def __init__(self):
        # 动漫人物图像的像素大小
        self.width = 96
        self.height = 96
        self.channels = 3
        self.shape = (self.width, self.height, self.channels)

        # 噪声采样的大小
        self.noise_dim = 100

        self.OPTIMIZER = optimizers.Adam(lr=0.0002)

        # 构建generator和discriminator
        self.generator = self.generator()

        self.discriminator = self.discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])

        # 因为完整模型是用来对generator进行训练的
        # 所以需要将discriminator的参数固定起来
        self.discriminator.trainable = False

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
        model.add(Reshape(self.shape))

        model.summary()

        return model

    def discriminator(self):
        """Discriminate whether a avatar is real.
        """
        model = Sequential()

        model.add(Flatten(input_shape=self.shape))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.02))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.02))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.02))
        model.add(Dense(1))
        model.add(Activation(activation='sigmoid'))

        model.summary()

        return model

    def stacked_G_D(self):
        model = Sequential()

        model.add(self.generator)
        model.add(self.discriminator)

        return model

    def train(self, data_round, train_data, epochs, batch_size=128, save_interval=50):
        """Train the generative adversarial model to generate anime avatar.
        """
        for epoch in range(epochs):
            #* 训练discriminator
            random_idx = np.random.randint(0, train_data.shape[0], batch_size)
            real_img = train_data[random_idx]
            real_label = np.ones((batch_size,1))

            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            fake_img = self.generator.predict(noise)
            fake_label = np.zeros((batch_size,1))

            d_loss_real = self.discriminator.train_on_batch(real_img, real_label)
            d_loss_fake = self.discriminator.train_on_batch(fake_img, fake_label)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #* 训练generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            confusion_label = np.ones((batch_size, 1))

            g_loss = self.stacked.train_on_batch(noise, confusion_label)

            #* 输出训练信息
            print("data round: %2d epoch: %5d, [D loss: %2.2f] [D acc: %3.2f%%] [G loss: %2.2f]" % (data_round+1, epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_sample(data_round, epoch)

    def save_sample(self, data_round, epoch):
        """
        """
        # canvas size
        row, column = 3,3
        # 生成展示图片
        noise = np.random.normal(0, 1, (row * column, self.noise_dim))
        gen_img = self.generator.predict(noise)

        # 将像素值调整到 0-1
        gen_img = 0.5 * gen_img + 0.5

        fig, axs = plt.subplots(row, column)
        cnt = 0
        for r in range(row):
            for c in range(column):
                # axs[r,c].imshow(gen_img[cnt,:,:,0], cmap='gray')
                axs[r,c].imshow(gen_img[cnt])
                axs[r,c].axis('off')
                cnt += 1
        fig.savefig('/home/xingyun/gan/images/data_round%d_%d.png' % (data_round, epoch))
        # fig.savefig('images\\g_%d.png' % epoch)
        plt.close()

if __name__ == '__main__':
    anime_gan = anime_GAN()
    
    for data_round in range(10):
        avatar_data = np.load(r'/data0/xingyun/anime_avatar/per_5000/avatar_data_' + str(5000*(data_round+1)) + '.npy')
        # avatar_data = np.load(r'/data0/xingyun/anime_avatar/avatar_data.npy')
        # avatar_data = np.load(r'F:\DATASETS\anime_avatar\avatar_data_10000.npy')

        anime_gan.train(data_round=data_round, train_data=avatar_data, epochs=10000, batch_size=32, save_interval=500)