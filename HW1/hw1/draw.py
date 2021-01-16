import numpy as np
import load_mnist as lm
from matplotlib import pyplot as plt

dataset = lm.load_mnist()
x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']

data0 = x_train[0]
data0 = np.array(data0, dtype='float')

data1 = x_train[5]
data1 = np.array(data1, dtype='float')

c = y_train.sum(axis=0)
print(c)

img_label = np.concatenate((x_train, y_train), axis=1) # (60000, 794)
# [:783] = img 
# [784:] = label
img0 = np.sum(element for element in img_label if element[:][784]==1)[0:784]
print(img0)
img0 /= c[0]
print(img0)

#print(img0.shape)
img0 = img0.reshape(28, 28)
plt.imshow(img0, cmap='gray')

#
#pixels_1 = data0.reshape(28, 28)
#pixels_2 = data1.reshape(28, 28)
#
#sum = pixels_1 + pixels_2
#pixel_x = sum.reshape(28, 28)
#print(pixels_1)
#print(pixels_2)
#print(pixel_x)
#plt.imshow(pixels_1, cmap='gray')
#fig1 = plt.imshow(pixels_2, cmap='gray')
#fig2 = plt.imshow(pixel_x, cmap='gray')
#img = Image.fromarray(data0, 'F')
plt.show()

