import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from random import shuffle
import tensorflow_datasets as tfds
import tensorflow as tf
import  math

data = np.array(range(10)).repeat(10) * 10
labels = np.array(range(10)).repeat(10)

data = list(zip(data, labels))
shuffle(data)


def load_hist():
    train, info = tfds.load(
        "colorectal_histology",
        split=["train"],
        shuffle_files=True,
        as_supervised=False,
        with_info=True
    )
    tfds.show_examples(train[0], info, rows=4, cols=4)

    train = train[0]

    data = []
    for image in train:
        data.append((image["image"].numpy()/255.0, int(image["label"].numpy())))

    shuffle(data)
    train_data = data[:4000]
    test_data = data[4000:]

    return train_data, test_data




def generate_distribution(n_clients: int, dispersion:int=1):
    """ This function generates a list of :n_clients elements. Each element in this list represents a percentage
    of how many of the given data is being put to the client. The percentages are calculated using the dirichlet
    distribution.

    Parameters:
    ------------
        n_clients : int
            This is the number of clients. The size of the returned list is the same as n_clients.
        dispersion : int
            This number indicates how much of a difference are between the percentages. A high number indicates a higher
            difference between the clients. The default value is set to 1.
            NOTE: 1 does NOT mean an equal distribution.

    Returns:
    ----------
    distribution : list
        A list of percentage values that add up to 1.
    """
    import numpy as np, numpy.random
    distribution = np.random.dirichlet(np.ones(n_clients) / dispersion, size=1)
    return distribution[0]


def split_by_distribution(number_of_data: int, distribution: list):
    """This function calculates the real number of data that each client gets, given the distribution list from the
    generate distribution method and the (amount of) data noted n. The function iterates through the distributions,
    multiplies it by n and floors the result in order to avoid distributing more data than available. The rest is
    being split up randomly over the clients.

    Parameters:
    --------------
        number_of_data : int
            Amount of data
        dispersion : list
            List of each percentage of the clients
    Returns:
    ------------
     amount_of_data : list
        list of integer values. Each element represents the natural number of data each client gets.
    """
    from math import floor
    import random
    amount_of_data = [floor(percentage * number_of_data) for percentage in distribution]
    if sum(amount_of_data) != number_of_data:
        delta = number_of_data - sum(amount_of_data)
        for _ in range(delta):
            random_index = random.randint(0, len(distribution)-1)
            amount_of_data[random_index] += 1

    return amount_of_data


def split_data_by_number_list(data, dispersion_list):
    data_splits = []
    x_shuffled, y_shuffled = zip(*data)
    start_index = 0
    for client_amount in dispersion_list:
        end_index = start_index + client_amount
        print(client_amount, start_index, end_index)
        if end_index <= len(x_shuffled):
            x_split = x_shuffled[start_index:end_index]
            y_split = y_shuffled[start_index:end_index]
            data_splits += [list(zip(x_split, y_split))]
            start_index = end_index  # the end of a list is exclusive
    return data_splits


def split_by_label(data, n_clients):
    """ This function gets the data as a zip object with (x_train, y_train) order. It then creates a copy of this
    in order to iterate through it. (Zip is an iterator so it is deleted when called)
    Then it is checked if the labels are raw (np.uint8) or in a list ([6]). If yes the labels are unpacked. Then the
    labels are put as a set in order to find out the unique values and the dictionary is created with all keys.
    Then one iterates through the whole data and sorts the "image" to its specific key in the sorted dictionary.
    Then the dictionary is "unpacked" which means that tuples are created (image, label). As a result
    there is a list of 10 different lists each containing data from one label.
    After that each label list is split into halves and each client is getting 1 half of 2 different labels.
    Then the halves of each client are merged and shuffeled.

     Parameters:
    --------------
        data : zip object
           The whole image data zipped with its label

    Returns:
    ------------
        A list of lists with images
    """
    import copy
    iterating_data = copy.deepcopy(data)
    all_data, all_labels = zip(*data)

    #  check if labels in array or not
    parse_labels = False
    if isinstance(all_labels[0], np.ndarray):
        parse_labels = True
        all_labels = [a[0] for a in all_labels]

    # dictionary each key has its images
    label_set = set(all_labels)
    sorted_data = {key: [] for key in label_set}
    for single_data, single_label in iterating_data:
        if not isinstance(single_label, np.uint8) and len(single_label) == 1:
            single_label = single_label[0]
        sorted_data[single_label].append(single_data)

    #  dictionary is unpacked and lists are split into halves
    sorted_list = []
    for every_key in sorted_data:
        one_label = []
        for data_point in sorted_data[every_key]:
            if parse_labels:
                one_label.append((data_point, [every_key]))
            else:
                one_label.append((data_point, every_key))
        first = one_label[:len(one_label) // 2]
        second = one_label[len(one_label) // 2:]
        sorted_list.append([first, second])

    #  clients get their data
    distributed_to_clients = [[] for i in range(n_clients)]
    for index, list in enumerate(sorted_list):
        distributed_to_clients[index].append(list.pop())
        if index+1 == len(sorted_list):
            distributed_to_clients[0].append(sorted_list[-1].pop())
        else:
            distributed_to_clients[index+1].append(list.pop())

    #  different labels are being joined and data gets shuffled
    for index, client in enumerate(distributed_to_clients):
        joined = client[0] + client[1]
        shuffle(joined)
        distributed_to_clients[index] = joined

    return distributed_to_clients


if __name__ == '__main__':
    load_hist()
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    mnist_train = zip(x_train, y_train)

    dist = generate_distribution(10, dispersion=0.01)
    amount = split_by_distribution(number_of_data=len(y_train), distribution=dist)
    set = split_by_label(mnist_train, 10)
    """

