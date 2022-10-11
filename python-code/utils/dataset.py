"""
    Copyright (C) 2019 Adrian Edward Thomas Henkel, Reza NasiriGerdeh

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    Author: Adrian Edward Thomas Henkel
"""
import tensorflow_datasets as tfds
import numpy as np
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils.np_utils import to_categorical
from random import shuffle


from utils.dataset_type import DatasetType
from utils.config_reader import ConfigReader


class Dataset:
    def __init__(self, config: ConfigReader):
        self.path = config.data["dataset_path"]
        self.name = config.data["dataset_name"]
        self.data_shape = config.data["image_shape"]
        self.data_type = config.data["data_type"]
        if "non_iid" in config.data:
            self.non_iid = config.data["non_iid"]
        else:
            self.non_iid = False
        if config.data["dataset_type"] == "c":
            self.type = DatasetType.COLOR_IMAGE
        elif config.data["dataset_type"] == "g":
            self.type = DatasetType.GRAYSCALE_IMAGE
        else:
            raise ValueError("Invalid config input for dataset_type")

        self.train_data = np.array([])
        self.__train_data_x = np.array([])
        self.__train_data_y = np.array([])
        self.train_data_size = -1

        self.test_data = np.array([])
        self.__test_data_x = np.array([])
        self.__test_data_y = np.array([])
        self.test_data_size = -1

    def open(self):
        """Opens the dataset given in the config file. The data is split into train (60k) and test data (10k).
        The train data is zipped into the class variable train_data. The length of the train and
        test data is saved."""
        print(f"Opening {self.name} dataset ...")
        if self.name == "mnist":
            (
                (self.__train_data_x, self.__train_data_y),
                (self.__test_data_x, self.__test_data_y),
            ) = mnist.load_data()
            self.train_data = list(zip(self.__train_data_x, self.__train_data_y))

            self.train_data_size = len(self.__train_data_x)
            self.test_data_size = len(self.__test_data_x)
        elif self.name == "fashion_mnist":
            (
                (self.__train_data_x, self.__train_data_y),
                (self.__test_data_x, self.__test_data_y),
            ) = fashion_mnist.load_data()
            self.train_data = list(zip(self.__train_data_x, self.__train_data_y))

            self.train_data_size = len(self.__train_data_x)
            self.test_data_size = len(self.__test_data_x)
        elif self.name == "cifar10":
            (
                (self.__train_data_x, self.__train_data_y),
                (self.__test_data_x, self.__test_data_y),
            ) = cifar10.load_data()

            self.train_data = list(zip(self.__train_data_x, self.__train_data_y))

            self.train_data_size = len(self.__train_data_x)
            self.test_data_size = len(self.__test_data_x)
        elif self.name == "colorectal":
            train, info = tfds.load(
                "colorectal_histology",
                split=["train"],
                shuffle_files=True,
                as_supervised=False,
                with_info=True,
            )
            # tfds.show_examples(train[0], info, rows=4, cols=4)
            train = train[0]
            data = []
            for image in train:
                data.append((image["image"].numpy(), int(image["label"].numpy())))

            shuffle(data)
            self.__train_data_x = np.array(
                [np.array(img_label[0]) for img_label in data[:4000]]
            )
            self.__train_data_y = np.array(
                [img_label[1] for img_label in data[:4000]], dtype=np.uint8
            )

            self.__test_data_x = np.array(
                [np.array(img_label[0]) for img_label in data[4000:]]
            )
            self.__test_data_y = np.array(
                [img_label[1] for img_label in data[4000:]], dtype=np.uint8
            )

            self.train_data = list(zip(self.__train_data_x, self.__train_data_y))
            self.train_data_size = len(self.__train_data_x)
            self.test_data_size = len(self.__test_data_x)
        else:
            raise RuntimeError("Bad dataset_name in config file")

    def preprocess(self, max_value: int, label_count: int):
        """The data is converted into the data type given in the config. (float32 mostly).
        Afterwards the data is scaled to a number between 0 and 1 by dividing by the
        maximum number (255 when images are used).
        The data is reshaped to specifiy how many channels are used (greyscale 1 channel,
        color 3 channels). -> e.g. (60000, 28, 28) to (60000, 28, 28, 1)
        __train_data_y is changed from and ndarray containing only the correct numbers (from 0-9)
        to an array of arrays of size ten where the index of the one indicates the corrext number.
        e.g. 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        Parameters:
        max_value : int
            The maximal value within the image matrices
        label_count: int
            The number of trained and predicted labels. This is necessary in order to split the data
            equally.
        """
        print(f"Preprocessing {self.name} dataset ...")

        # X train and test data
        self.__train_data_x = self.__train_data_x.astype(self.data_type)
        self.__train_data_x /= max_value

        self.__test_data_x = self.__test_data_x.astype(self.data_type)
        self.__test_data_x /= max_value

        if self.type == DatasetType.GRAYSCALE_IMAGE:
            shape_train = (
                self.__train_data_x.shape[0],  # 60000
                self.data_shape[0],  # 28
                self.data_shape[1],  # 28
                1,
            )
            shape_test = (
                self.__test_data_x.shape[0],
                self.data_shape[0],
                self.data_shape[1],
                1,
            )
        elif self.type == DatasetType.COLOR_IMAGE:
            shape_train = (
                self.__train_data_x.shape[0],
                self.data_shape[0],
                self.data_shape[1],
                3,
            )
            shape_test = (
                self.__test_data_x.shape[0],
                self.data_shape[0],
                self.data_shape[1],
                3,
            )
        else:
            print(f"{self.type} is an invalid dataset type!")
            return

        self.__train_data_x = self.__train_data_x.reshape(shape_train)
        self.__test_data_x = self.__test_data_x.reshape(shape_test)

        # Y train data to onehot
        self.__train_data_y = np.eye(label_count)[self.__train_data_y]

        # Reformat the Y data from (n, 1, 10) into (n, 10)
        if len(self.__train_data_y.shape) == 3:
            print("Reformatting Y data")
            self.__train_data_y = np.reshape(
                self.__train_data_y, (self.__train_data_y.shape[0], label_count)
            )

        # bundle X and Y to a tuple
        self.train_data = list(zip(self.__train_data_x, self.__train_data_y))
        self.test_data = (self.__test_data_x, self.__test_data_y)

    def get_batch(self, batch_size):
        """Returns a batch of data for the training.

        Parameters
        ----------
        batch_size: int
                the size of the batch

        Returns:
        --------
        x_batch : 3-dim np-array
            Image data
        y_batch : 2-dim np-array
            Image label
        """
        print(f"Getting batch of size {batch_size} ...")
        np.random.shuffle(self.train_data)
        x_shuffled, y_shuffled = zip(*self.train_data)

        batch_count = int(np.ceil(len(x_shuffled) / batch_size))

        x_batch = np.array_split(x_shuffled, batch_count)
        y_batch = np.array_split(y_shuffled, batch_count)

        return x_batch, y_batch

    def split_train_data(self, split_count):
        """Splits the train data into equal sets if non iid.
        Parameters:
        -----------
        split_count : int
            The number of sets. For the simulation the number of clients is used.

        Returns:
        --------
        data_splits
            A collection of data that is later given to the Clients
        """
        print(f"Splitting the train dataset into {split_count} partitions ...")

        np.random.shuffle(self.train_data)

        if self.non_iid:
            return self.split_by_label(split_count)

        x_shuffled, y_shuffled = zip(*self.train_data)

        data_splits = []

        partition_size = self.train_data_size // split_count

        for start_index in np.arange(0, self.train_data_size, partition_size):
            end_index = start_index + partition_size
            if end_index <= self.train_data_size:
                x_split = x_shuffled[start_index:end_index]
                y_split = y_shuffled[start_index:end_index]
                data_splits += [list(zip(x_split, y_split))]

        return data_splits

    def split_by_label(self, n_clients):
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
        print("The data is being split non IID to the clients.")
        import copy

        iterating_data = copy.deepcopy(self.train_data)
        all_data, all_labels = zip(*self.train_data)

        #  check if labels in array or not
        label_in_list = False
        label_as_onehot = False
        if isinstance(all_labels[0], np.ndarray):
            if len(all_labels[0]) == 1:
                label_in_list = True
                all_labels = [a[0] for a in all_labels]
            elif len(all_labels[0]) > 1:
                label_as_onehot = True
                all_labels = [
                    np.where(single_label == 1.0)[0][0] for single_label in all_labels
                ]

        # dictionary each key has its images
        label_set = set(all_labels)
        sorted_data = {key: [] for key in label_set}
        for single_data, single_label in iterating_data:
            if not isinstance(single_label, np.uint8):
                if label_in_list:
                    single_label = single_label[0]
                elif label_as_onehot:
                    single_label = np.where(single_label == 1.0)[0][0]

            sorted_data[single_label].append(single_data)

        #  dictionary is unpacked and lists are split into halves
        sorted_list = []
        for every_key in sorted_data:
            one_label = []
            for data_point in sorted_data[every_key]:
                if label_in_list:
                    one_label.append((data_point, [every_key]))
                elif label_as_onehot:
                    tmp_label = np.zeros(n_clients, np.float)
                    tmp_label[every_key] = 1.0
                    one_label.append((data_point, tmp_label))
                else:
                    one_label.append((data_point, every_key))
            first = one_label[: len(one_label) // 2]
            second = one_label[len(one_label) // 2 :]
            sorted_list.append([first, second])

        #  clients get their data
        distributed_to_clients = [[] for i in range(n_clients)]
        for index, list in enumerate(sorted_list):
            distributed_to_clients[index].append(
                list.pop()
            )  # client n gets data of label n
            if index + 1 == len(sorted_list):  # last client gets data of first label
                distributed_to_clients[0].append(sorted_list[-1].pop())
            else:
                distributed_to_clients[index + 1].append(
                    list.pop()
                )  # client n gets data of label n+1

        #  different labels are being joined and data gets shuffled
        for index, client in enumerate(distributed_to_clients):
            joined = client[0] + client[1]
            shuffle(joined)
            distributed_to_clients[index] = joined

        return distributed_to_clients
