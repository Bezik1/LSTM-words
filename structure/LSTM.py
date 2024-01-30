from helpers.vocab_functions import initWeights, char_to_idx, idx_to_char, oneHotEncode, char_size
from helpers.functions import sigmoid, softmax, tanh, d_sigmoid, d_tanh
import numpy as np
from tqdm import tqdm
import re

# Long Short-Term Memory Network Class
class LSTM:
    def __init__(self, hyperparameters, loaded=False):
        # Hyperparameters
        self.learning_rate = hyperparameters.learning_rate
        self.hidden_size = hyperparameters.hidden_size
        self.epochs = hyperparameters.epochs
        self.input_size = hyperparameters.input_size
        self.output_size = hyperparameters.output_size

        if loaded:
            return
        
        # Forget Gate
        self.wf = initWeights(self.input_size, self.hidden_size)
        self.bf = np.zeros((self.hidden_size, 1))

        # Input Gate
        self.wi = initWeights(self.input_size, self.hidden_size)
        self.bi = np.zeros((self.hidden_size, 1))

        # Candidate Gate
        self.wc = initWeights(self.input_size, self.hidden_size)
        self.bc = np.zeros((self.hidden_size, 1))

        # Output Gate
        self.wo = initWeights(self.input_size, self.hidden_size)
        self.bo = np.zeros((self.hidden_size, 1))

        # Final Gate
        self.wy = initWeights(self.hidden_size, self.output_size)
        self.by = np.zeros((self.output_size, 1))

    # Reset Network Memory
    def reset(self):
        self.concat_inputs = {}

        self.hidden_states = {-1:np.zeros((self.hidden_size, 1))}
        self.cell_states = {-1:np.zeros((self.hidden_size, 1))}

        self.activation_outputs = {}
        self.candidate_gates = {}
        self.output_gates = {}
        self.forget_gates = {}
        self.input_gates = {}
        self.outputs = {}

    # Forward Propogation
    def forward(self, inputs):
        self.reset()

        outputs = []
        for q in range(len(inputs)):
            self.concat_inputs[q] = np.concatenate((self.hidden_states[q - 1], inputs[q]))

            self.forget_gates[q] = sigmoid(np.dot(self.wf, self.concat_inputs[q]) + self.bf)
            self.input_gates[q] = sigmoid(np.dot(self.wi, self.concat_inputs[q]) + self.bi)
            self.candidate_gates[q] = tanh(np.dot(self.wc, self.concat_inputs[q]) + self.bc)
            self.output_gates[q] = sigmoid(np.dot(self.wo, self.concat_inputs[q]) + self.bo)

            self.cell_states[q] = self.forget_gates[q] * self.cell_states[q - 1] + self.input_gates[q] * self.candidate_gates[q]
            self.hidden_states[q] = self.output_gates[q] * tanh(self.cell_states[q])

            outputs += [np.dot(self.wy, self.hidden_states[q]) + self.by]

        return outputs

    # Backward Propogation
    def backward(self, errors, inputs):
        d_wf, d_bf = 0, 0
        d_wi, d_bi = 0, 0
        d_wc, d_bc = 0, 0
        d_wo, d_bo = 0, 0
        d_wy, d_by = 0, 0

        dh_next, dc_next = np.zeros_like(self.hidden_states[0]), np.zeros_like(self.cell_states[0])
        for q in reversed(range(len(inputs))):
            error = errors[q]

            # Final Gate Weights and Biases Errors
            d_wy += np.dot(error, self.hidden_states[q].T)
            d_by += error

            # Hidden State Error
            d_hs = np.dot(self.wy.T, error) + dh_next

            # Output Gate Weights and Biases Errors
            d_o = tanh(self.cell_states[q]) * d_hs * d_sigmoid(self.output_gates[q])
            d_wo += np.dot(d_o, inputs[q].T)
            d_bo += d_o

            # Cell State Error
            d_cs = d_tanh(tanh(self.cell_states[q])) * self.output_gates[q] * d_hs + dc_next

            # Forget Gate Weights and Biases Errors
            d_f = d_cs * self.cell_states[q - 1] * d_sigmoid(self.forget_gates[q])
            d_wf += np.dot(d_f, inputs[q].T)
            d_bf += d_f

            # Input Gate Weights and Biases Errors
            d_i = d_cs * self.candidate_gates[q] * d_sigmoid(self.input_gates[q])
            d_wi += np.dot(d_i, inputs[q].T)
            d_bi += d_i
            
            # Candidate Gate Weights and Biases Errors
            d_c = d_cs * self.input_gates[q] * d_tanh(self.candidate_gates[q])
            d_wc += np.dot(d_c, inputs[q].T)
            d_bc += d_c

            # Concatenated Input Error (Sum of Error at Each Gate!)
            d_z = np.dot(self.wf.T, d_f) + np.dot(self.wi.T, d_i) + np.dot(self.wc.T, d_c) + np.dot(self.wo.T, d_o)

            # Error of Hidden State and Cell State at Next Time Step
            dh_next = d_z[:self.hidden_size, :]
            dc_next = self.forget_gates[q] * d_cs

        for d_ in (d_wf, d_bf, d_wi, d_bi, d_wc, d_bc, d_wo, d_bo, d_wy, d_by):
            np.clip(d_, -1, 1, out = d_)

        self.wf += d_wf * self.learning_rate
        self.bf += d_bf * self.learning_rate

        self.wi += d_wi * self.learning_rate
        self.bi += d_bi * self.learning_rate

        self.wc += d_wc * self.learning_rate
        self.bc += d_bc * self.learning_rate

        self.wo += d_wo * self.learning_rate
        self.bo += d_bo * self.learning_rate

        self.wy += d_wy * self.learning_rate
        self.by += d_by * self.learning_rate

    # Train
    def train(self, train_x, train_y):
        accuracy_history = []

        for _ in tqdm(range(self.epochs)):
            for inputs, labels in zip(train_x, train_y):
                encoded_inputs = [oneHotEncode(input) for input in inputs]
                predictions = self.forward(encoded_inputs)

                errors = []
                for q in range(len(predictions)):
                    errors += [-softmax(predictions[q])]
                    errors[-1][char_to_idx[labels[q]]] += 1

                self.backward(errors, self.concat_inputs)

            if _ % 2 == 0:
                accuracy = 0
                for inputs, labels in zip(train_x, train_y):
                    accuracy += self.test(inputs, labels, False)
                accuracy_history.append(round(accuracy / len(train_x), 2))
                print(f'Epoch: {_} | Accuracy: {round(accuracy / len(train_x), 2)}%')

        return accuracy_history
    
    # Test
    def test(self, inputs, labels, show):
        accuracy = 0
        probabilities = self.forward([oneHotEncode(input) for input in inputs])
        labels_clear = [t for t in labels if t != "0"]
        output = ''

        for q in range(len(probabilities)):
            prediction = idx_to_char[np.argmax(softmax(probabilities[q].reshape(-1)))]
            output += prediction
            if prediction == "0" and labels[q] == "0":
                continue
            elif prediction == labels[q]:
                accuracy += 1

        question = [t for t in inputs if t != "0"]
        answer = [t for t in output if t != "0"]
        accuracy_percent = round(accuracy * 100 / (len(labels_clear)+0.000001), 2)
        if show:
            print(f'Question:\n{question}\n')
            print(f'Predictions:\n{"".join(answer)}\n')
            print(f'Accuracy: {accuracy_percent}%')

        return accuracy_percent

    # Dev Tools
    def save(self, filename):
        np.savez(filename,
            wf=self.wf, 
            bf=self.bf,
            wi=self.wi, 
            bi=self.bi,
            wc=self.wc, 
            bc=self.bc,
            wo=self.wo, 
            bo=self.bo,
            wy=self.wy, 
            by=self.by
        )

    # Dev Tools
    def load(self, filename):
        loaded_data = np.load(filename)
        self.wf = loaded_data['wf']
        self.bf = loaded_data['bf']
        self.wi = loaded_data['wi']
        self.bi = loaded_data['bi']
        self.wc = loaded_data['wc']
        self.bc = loaded_data['bc']
        self.wo = loaded_data['wo']
        self.bo = loaded_data['bo']
        self.wy = loaded_data['wy']
        self.by = loaded_data['by']