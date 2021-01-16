import numpy as np
import neuralnet as nl
import load_mnist as lm
np.random.seed(21)

dataset = lm.load_mnist()
x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']

img = np.zeros(28 * 28 * 10).reshape(10, 784)
img_test = np.zeros(28 * 28 * 10).reshape(10, 784)

c = np.zeros(10)
c_test = np.zeros(10)

# count labels
c = y_train.sum(axis=0)
c_test = y_test.sum(axis=0)

# x_train(60000, 784) concat y_train (60000, 10)
# (60000, 794)
# [:784] img (0~783)
# [784:] label (784~793)
img_label = np.concatenate((x_train, y_train), axis=1) # (60000, 794)
img_label_test = np.concatenate((x_test, y_test), axis=1) # (10000, 794)


# if we want the img array of number 0 we take arrays which [:][784] = 1
# and dvsn with the # of num 0 in the dataset
for i in range(10):
    img[i] = np.sum(element for element in img_label if element[:][784+i]==1)[:784]/c[i]    
    img_test[i] = np.sum(e for e in img_label_test if e[:][784+i]==1)[:784]/c_test[i]


np.set_printoptions(linewidth=125)
print()
print(np.around(img[0].reshape(28,28), 1))


import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300
plt.figure(1)

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.axis('off')
    plt.imshow(img[i].reshape(28,28), cmap='gray')
    # plt.show()


plt.figure(2)
x = np.arange(10)
plt.bar(x, c)
plt.xticks(x)
plt.yticks( np.arange(0, 7100, 1000) )
plt.show()


plt.figure(3)

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.axis('off')
    plt.imshow(img_test[i].reshape(28,28), cmap='gray')
    # plt.show()


plt.figure(4)

x = np.arange(10)
plt.bar(x, c_test)
plt.xticks(x)
plt.yticks( np.arange(0, 7100, 1000) )
plt.show()


