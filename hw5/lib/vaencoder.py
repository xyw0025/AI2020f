# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import LambdaCallback

import numpy as np
import matplotlib.pyplot as plt
import os

# network parameters
input_shape = (28, 28, 1)
        
input_shape = input_shape
batch_size = 128
kernel_size = 3
latent_dim = 2

# encoder/decoder number of CNN layers and filters per layer
layer_filters = [32, 64]

class Operation():
    def __init__(self):
        self.vaencoder = None
        return
    
    def loadData(self):
         # load MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # reshape to (: ,28, 28, 1) and normalize input images
        image_size = x_train.shape[1]
        
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        return [x_train, y_train, x_test, y_test]
    
    def construct(self):
        # build the vaencoder model
        encoder, shape, log, mean = EncoderModel.getEncoder(input_shape, layer_filters, latent_dim, kernel_size)
        decoder = EncoderModel.getDecoder(latent_dim, shape, layer_filters[::-1],  kernel_size)
        
        # combine encoder & decder inside vae 
        models = [encoder, decoder]
        
        self.vaencoder = EncoderModel.getVAencoder(models, log, mean)
    
    def train(self, train_data, test_data, output, epoch_size=1):
        
        x_train, y_train = train_data
        x_test, y_test = test_data
        
            
        batch_len = x_train.shape[0]/batch_size
        
        # callback function definition
        call = LambdaCallback(on_epoch_end=lambda epoch, logs: output(epoch=(epoch+1)/epoch_size, logs=logs),
                              on_batch_end=lambda batch, logs: output(batch=(batch+1)/batch_len, logs=logs))
        
        self.vaencoder.fit(x_train, x_train, 
                             epochs=epoch_size, 
                             validation_data=(x_test, x_test),
                             batch_size=batch_size,
                             callbacks=[call]
                             )

    def predict(self, x):
        return self.vaencoder.predict(x)
    
    def encode(self, x):
        encoder = self.vaencoder.get_layer('encoder')
        return encoder.predict(x)[2][0]
    
    def decode(self, x, y):
        z = np.array([[x, y]])
        decoder = self.vaencoder.get_layer('decoder')
        x_decoded = decoder.predict(z)
        return x_decoded
        
    def prepared(self):
        if self.vaencoder == None:
            return False
        return True
        
    def save(self, path = 'models'):
        print('save models')
        path = os.path.splitext(path)[0]
        #path = os.path.join(path, 'vae_cnn_mnist.tf')
        self.vaencoder.save_weights(path)
        #save_model(self.vaencoder, 'models', save_format="tf")
            
    def load(self, path = 'models'):
        print('load models')
        
        print(path)
        if os.path.isfile(path):
            #path = os.path.join(path, 'vae_cnn_mnist.tf')
            path = os.path.splitext(path)[0]
            self.vaencoder.load_weights(path)
            #self.vaencoder = load_model('models')
            return True
        else:
            return False
    
    def plot(self, x_test, y_test):
        data = [x_test, y_test]
        encoder = self.vaencoder.get_layer('encoder')
        decoder = self.vaencoder.get_layer('decoder')
        #fig_en = EncoderModel.plot_latent_2dim(encoder, data, batch_size)
        fig_de = EncoderModel.plot_digits_over_latent(decoder)
        return None, fig_de
        

class EncoderModel:
    
    def plot_latent_2dim(encoder, data, batch_size):
        x_test, y_test = data
        xmin = ymin = -4
        xmax = ymax = +4
        
        # display a 2D plot of the digit classes in the latent space
        z = encoder.predict(x_test,
                            batch_size=batch_size)[2]
        fig = plt.figure(figsize=(10, 10))
    
        # axes x and y ranges
        axes = plt.gca()
        axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
    
        # subsample to reduce density of points on the plot
        z = z[0::2]
        y_test = y_test[0::2]
        plt.scatter(z[:, 0], z[:, 1], marker="")
        for i, digit in enumerate(y_test):
            axes.annotate(digit, (z[i, 0], z[i, 1]))
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()

        return fig
        
    def plot_digits_over_latent(decoder):
        xmin = ymin = -4
        xmax = ymax = +4

        # display a 30x30 2D manifold of the digits
        n = 10
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(xmin, xmax, n)
        grid_y = np.linspace(ymin, ymax, n)[::-1]
    
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = np.array([[xi, yi]])
                x_decoded = decoder.predict(z)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
    
        fig = plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.imshow(figure, cmap='Greys_r')
        plt.show()
        return fig
    
    def getEncoder(input_shape, layer_filters, latent_dim, kernel_size):
        # reparameterization trick
        # instead of sampling from Q(z|X), sample eps = N(0,I)
        # then z = z_mean + sqrt(var)*eps
        def sampling(args):
            """Reparameterization trick by sampling 
                fr an isotropic unit Gaussian.
        
            # Arguments:
                args (tensor): mean(z_mean) and log(z_log_var) of variance of Q(z|X)
        
            # Returns:
                z (tensor): sampled latent vector
            """
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean=0 and std=1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        
        inputs = Input(shape=input_shape, name="encoder_input")
        x = inputs
        for filters in layer_filters:
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same', activation='relu')(x)
        
        shape = K.int_shape(x)
        
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        
        encoder = Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        plot_model(encoder, to_file='vae_encoder_plot.png', show_shapes=True)
        
        return encoder, shape, z_log_var, z_mean
        
 
    
    def getDecoder(latent_dim, shape, layer_filters, kernel_size):
        
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3],
                            activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        
        for filters in layer_filters:
            x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding='same', activation='relu')(x)
        
        outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size, padding='same', activation='sigmoid', name='decoder_output')(x)
        
        
        decoder = Model(inputs=latent_inputs, outputs=outputs, name='decoder')
        plot_model(decoder, to_file='vae_decoder_plot.png', show_shapes=True)
        
        decoder.summary()
        
        return decoder
        
        
        
    def getVAencoder(models, z_log_var, z_mean):
        encoder, decoder = models
        
        
        #輸入層
        inputs = encoder.input 
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs=inputs, outputs=outputs, name='vae')
        vae.summary()
        
        
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        reconstruction_loss *= 28 * 28
        
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')
        return vae
    