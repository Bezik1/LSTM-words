from helpers.Hyperparameters import Hyperparameters
from helpers.Visualizer import Visualizer
from helpers.vocab_functions import oneHotEncode, idx_to_char
from helpers.functions import softmax
from const.paths import HYPERPARAMETERS_PATH, TRAINED_NETWORK
from const.visualizer import VISUALIZER_COLUMNS, VISUALIZER_SIZE
from structure.LSTM import LSTM
from data.data import train_x, train_y, convert_to_same_len, DATA_SIZE, SPECIAL_SIGNS, chars
import numpy as np
import re

#print(train_x)
#print(sorted(chars))
#print("reagujesz." in chars)
hyperparameters = Hyperparameters(HYPERPARAMETERS_PATH)

lstm = LSTM(hyperparameters)
accuracy_history = lstm.train(train_x, train_y)
lstm.save(TRAINED_NETWORK)

for inputs, labels in zip(train_x, train_y):
    lstm.test(inputs, labels, False)

visualizer = Visualizer(VISUALIZER_SIZE, VISUALIZER_COLUMNS)

visualizer.draw(
    [(accuracy_history, "Accuracy")],
    "Epoch",
    "Percent",
    "Accuracy"
)

visualizer.visualize()

while True:
    question = input("User: ") + " ".lower()
    question = re.sub(f'[{SPECIAL_SIGNS}]', r' ', question)
    question = convert_to_same_len([t for t in question.split(" ") if t != ""], DATA_SIZE)

    encoded_question = [oneHotEncode(input) for input in question]
    probabilities = lstm.forward(encoded_question)

    output = ' '
    for q in range(len(question)):
        prediction = idx_to_char[np.argmax(softmax(probabilities[q].reshape(-1)))]
        output += f"{prediction} "

    answer = re.sub(r'0', '', output)
    print(f'Bezik: {answer}')