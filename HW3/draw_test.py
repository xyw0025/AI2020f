#!/usr/bin/env python3

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
print(y_train[0])
pixels_0 = data0.reshape(28, 28)
plt.imshow(pixels_0, cmap='gray')
plt.show()

