from helpers.Hyperparameters import Hyperparameters
from const.paths import HYPERPARAMETERS_PATH, TRAINED_NETWORK
from structure.LSTM import LSTM
from data.data import train_x, train_y

hyperparameters = Hyperparameters(HYPERPARAMETERS_PATH)

lstm = LSTM(hyperparameters, True)
lstm.load(TRAINED_NETWORK)
for inputs, labels in zip(train_x, train_y):
    lstm.test(inputs, labels, True)