import numpy as np

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def d_sigmoid(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - x ** 2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0))
    softmax_result = exp_x / np.sum(exp_x, axis=0)
    softmax_result[np.isnan(softmax_result)] = 0
    return softmax_result