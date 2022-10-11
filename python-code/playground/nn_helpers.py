import numpy
import matplotlib.pyplot as plt


def visualize_matrix(matrix: numpy.ndarray) -> None:
    plt.imshow(matrix)
    plt.show()
    plt.close()
