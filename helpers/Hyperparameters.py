import json

# Hyperparameters of machine learning neural network model
class Hyperparameters:
    def __init__(self, path):
        self.path = path
        self.load_data()

    # Loading data
    def load_data(self):
        try:
            file = open(self.path, "r")
            data = json.load(file)

            self.learning_rate = float(data['learning_rate'])
            self.epochs = int(data['epochs'])
            self.input_size = int(data['input_size'])
            self.hidden_size = int(data['hidden_size'])
            self.output_size = int(data['output_size'])
        
        except FileNotFoundError:
            print("Hyperparameters file not found.")
            exit(1)
        
        except (KeyError, ValueError):
            print("Invalid or missing data in hyperparameters file.")
            exit(1)