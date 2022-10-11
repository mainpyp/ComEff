import sys
import numpy as np
from random import randint
from tensorflow.keras.datasets import mnist, fashion_mnist, imdb
from sklearn.preprocessing import MinMaxScaler


def get_simulated_data(scaled: bool) -> tuple:
    """ Simulates some numerical data in terms of a tutorial \n
    Arguments:
        :param scaled:  if True the data is being scaled to numbers between 0 and 1 using MinMax scaler.
                        If False the original values are taken
    Data that is being simulated:
        - drug was tested on individuals from ages 13 to 100
        - .95 of patients 65 or older had side effects
        - .95 of patients under 65 had no side effects
    """
    train_samples = []
    train_labels = []

    for i in range(950):  # 950 / 1000 -> 0.95 'normal'
        random_young = randint(13, 64)
        train_samples.append(random_young)
        train_labels.append(0)

        random_old = randint(65, 100)
        train_samples.append(random_old)
        train_labels.append(1)

    for i in range(50):  # 50 / 1000 -> 0.05 'outlayers'
        random_young = randint(13, 64)
        train_samples.append(random_young)
        train_labels.append(1)

        random_old = randint(65, 100)
        train_samples.append(random_old)
        train_labels.append(0)

    train_labels = np.array(train_labels)
    train_samples = np.array(train_samples)

    """
    This section reshapes the data from to a range from 0 to 1 instead of 13 to 100
    .reshape(-1, 1) is just a formality since fit_transform is not accepting 1D np.array s
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_sample = scaler.fit_transform(train_samples.reshape(-1, 1))

    if scaled:
        return scaled_train_sample, train_labels

    return train_samples, train_labels


def get_mnist_data() -> tuple:
    """Loads the mnist data set and splits it into labeled train and test data

    :return: (sample_train_data, label_train_data), (sample_test_data, label_test_data)
    """
    print("[MNIST number] Loading dataset.")
    (sample_train, label_train), (sample_test, label_test) = mnist.load_data()
    print("[MNIST number] Done loading dataset.")

    sample_train, sample_test = sample_train / 255.0, sample_test / 255.0

    return sample_train, label_train, sample_test, label_test


if __name__ == "__main__":
    get_mnist_data()
    print("done")
