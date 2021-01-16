
import numpy as np

def sigmoid(x): 
    return 1/(1+np.exp(-x))

def inner_product(x_train, w, b):
    return np.dot(x_train, w) + b




def test(x_train, w_list, b_list):
    
#   x_train: (1, 784), w_list[0]: (784, 16), b_list[0]: (16,)
#    print(len(w_list))
    print(r"x_train: {}, w_list: {}, b_list: {}".format(x_train.shape, w_list[1].shape, b_list[1].shape))
#    print(r"x_train: {}, w_list: {}, b_list: {}", format(x_train.shape, w_list.shape, b_list.shape))

def calculate(x_train, w_list, b_list):
    
    val_dict = {}
    
    # (1, 784) dot (784, 16) 
    a_1 = inner_product(x_train, w_list[0], b_list[0]) # (1, 16)
    y_1 = sigmoid(a_1) # (N, 100)
    a_2 = inner_product(y_1, w_list[1], b_list[1]) # (N, 10)
    y_2 = sigmoid(a_2)

    
    a_3 = inner_product(y_2, w_list[2], b_list[2])
    y_3 = sigmoid(a_3)  # (1,10)
    
    y_3 /= np.sum(y_3, axis=1, keepdims=True)  
    
    
    
    val_dict['a_1'] = a_1
    val_dict['y_1'] = y_1
    val_dict['a_2'] = a_2
    val_dict['y_2'] = y_2
    val_dict['a_3'] = a_3
    val_dict['y_3'] = y_3
    
    return val_dict

