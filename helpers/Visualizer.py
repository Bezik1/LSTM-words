import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, size, columns) -> None:
        self.size = size
        self.columns = columns
        self.row = 1

    def draw(self, values_list, x_label, y_label, title):
        plt.subplot(self.size, self.columns, self.row)

        for values, label in values_list:
            plt.plot(values, label=label)
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        self.row += 1
    
    def visualize(self):
        plt.tight_layout()
        plt.show()