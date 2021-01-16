# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import LambdaCallback

import numpy as np
import matplotlib.pyplot as plt
import os

# network parameters
input_shape = (28, 28, 1)
        
input_shape = input_shape
batch_size = 32
kernel_size = 3
latent_dim = 2

# encoder/decoder number of CNN layers and filters per layer
layer_filters = [32, 64]

class Operation():
    
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
        # build the autoencoder model
        encoder, shape = EncoderModel.getEncoder(input_shape, layer_filters, latent_dim, kernel_size)
        decoder = EncoderModel.getDecoder(shape, latent_dim, layer_filters[::-1], kernel_size)
        
        models = [encoder, decoder]
        self.autoencoder = EncoderModel.getAutoencoder(models)
    
    def train(self, train_data, test_data, output, epoch_size=1):
        
        x_train, y_train = train_data
        x_test, y_test = test_data
        
        callback = LambdaCallback(on_epoch_end=lambda epoch, logs: output((epoch+1)/epoch_size))
 
        self.autoencoder.fit(x_train, y_train, 
                             epochs=epoch_size, 
                             validation_data=(x_test, y_test),
                             batch_size=batch_size,
                             callbacks=callback)

    def predict(self, x):
        return self.autoencoder.predict(x)
    
    def encode(self, x):
        encoder = self.autoencoder.get_layer('encoder')
        return encoder.predict(x)
    
    def decode(self, x, y):
        z = np.array([[x, y]])
        decoder = self.autoencoder.get_layer('decoder')
        x_decoded = decoder.predict(z)
        return x_decoded
    
    def prepared(self):
        if self.autoencoder is None:
            return False
        return True
        
    def save(self):
        self.autoencoder.save('models\\model.h5')
            
    def load(self):
        try:
            self.autoencoder = load_model('models\\model.h5')
            return True
        except OSError:
            return False
    
    def plot(self, x_test, y_test):
        data = [x_test, y_test]
        encoder = self.autoencoder.get_layer('encoder')
        decoder = self.autoencoder.get_layer('decoder')
        EncoderModel.plot_latent_2dim(encoder, data, batch_size)
        EncoderModel.plot_digits_over_latent(decoder)

class EncoderModel:
    
    def plot_latent_2dim(encoder, data, batch_size, model_name="autoencoder_2dim"):
        x_test, y_test = data
        xmin = ymin = -4
        xmax = ymax = +4
        
        os.makedirs(model_name, exist_ok=True)
    
        filename = os.path.join(model_name, "latent_2dim.png")
        # display a 2D plot of the digit classes in the latent space
        z = encoder.predict(x_test,
                            batch_size=batch_size)
        plt.figure(figsize=(12, 10))
    
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
        plt.savefig(filename)
        plt.show()
        
    def plot_digits_over_latent(decoder, model_name="autoencoder_2dim"):
        xmin = ymin = -4
        xmax = ymax = +4
        os.makedirs(model_name, exist_ok=True)
        filename = os.path.join(model_name, "digits_over_latent.png")
        # display a 30x30 2D manifold of the digits
        n = 30
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
    
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
        
    def getEncoder(input_shape, layer_filters, latent_dim, kernel_size):
        
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        # stack of Conv2D(32)-Conv2D(64)
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
            
        # shape info needed to build decoder model so we don't do hand computation
        # the input to the decoder's first Conv2DTranspose will have this shape
        # shape is (7, 7, 64) which is processed by the decoder back to (28, 28, 1)
        shape = K.int_shape(x)
        
        # generate latent vector
        x = Flatten()(x)
        latent = Dense(latent_dim, name='latent_vector')(x)
        
        # instantiate encoder model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()
#        plot_model(encoder, to_file='encoder.png', show_shapes=True)
        
        return encoder, shape

    
    def getDecoder(shape, latent_dim, layer_filters, kernel_size):
    
    
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        # use the shape (7, 7, 64) that was earlier saved
        x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
        # from vector to suitable shape for transposed conv
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        
        # stack of Conv2DTranspose(64)-Conv2DTranspose(32)
        for filters in layer_filters[::-1]:
                x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
            
        # reconstruct the input
        outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='decoder_output')(x)
        
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        
        return decoder
    
        
        
    def getAutoencoder(models):
        encoder, decoder = models
            
        # instantiate autoencoder model
        inputs = encoder.input
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()
        plot_model(autoencoder, to_file='img\\autoencoder.png', show_shapes=True)
        
        # Mean Square Error (MSE) loss function, Adam optimizer
        autoencoder.compile(loss='mse', optimizer='adam')
        return autoencoder
    