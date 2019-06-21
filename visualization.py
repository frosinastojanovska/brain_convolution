import pandas as pd
import matplotlib.pyplot as plt


def visualize_learning_loss(file_name, plot_file_name):
    file = pd.read_csv(file_name)
    lines = file.plot.line(x='epoch', y=['loss', 'val_loss'])
    plt.title('Learning curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(plot_file_name)
    plt.show()


if __name__ == '__main__':
    visualize_learning_loss('trained/logs/adhd.log', 'trained/logs/adhd_adamax.png')
