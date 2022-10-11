"""
    Copyright (C) 2020 Adrian Edward Thomas Henkel

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

import numpy as np
import json
import keras.backend as K

from benchmarking.logger import Logger

from utils.dataset import Dataset
from utils.dataset_type import DatasetType
from utils.config_reader import ConfigReader

from utils.model_builds.build_two_layer_percepron import build_two_layer_perceptron
from utils.model_builds.build_small_cnn_model import build_small_cnn_model
from utils.model_builds.build_tl_cnn_model import build_tl_cnn_model
from utils.model_builds.build_vgg16 import build_vgg16

from utils.model import DnnModel

from client_server.client import FederatedClient
from client_server.server import FederatedServer

from keras.optimizers import SGD

# TODO: when mlu is set reduce max iterations np.ceil(max_iterations / local_iterations)


class Simulator:
    def __init__(self, config):
        self.config = config
        self.logger = Logger(self.config)

        # open and prepare mnist dataset
        self.dataset = Dataset(
            config=self.config
        )

        self.mnist_train_data_splits = None
        self.plain_model = None
        self.model = None
        self.server: FederatedServer = None
        self.clients: list = []
        self.client_count: int = self.config.data["client_count"]

    def process_data(self):
        """Preprocesses the data. Dataset is opened, processed (scaled between 0 and 1, reshaped)
        and equally split"""
        self.dataset.open()

        self.dataset.preprocess(max_value=self.config.data["max_value"],
                                label_count=self.config.data["label_count"])

        self.mnist_train_data_splits = self.dataset.split_train_data(
            self.client_count
        )

    def setup_model(self):
        # build the model
        input_shape = self.dataset.train_data[0][0].shape
        label_count = self.config.data["label_count"]
        #label_count = self.mnist_dataset.train_data[0][1].shape[0]

        model_type = self.config.data["model_type"]
        if model_type == "tlp":
            self.plain_model = build_two_layer_perceptron(input_shape, label_count)
        elif model_type == "cnn":
            self.plain_model = build_small_cnn_model(input_shape, label_count)
        elif model_type == "tlcnn":
            self.plain_model = build_tl_cnn_model(input_shape, label_count)
        elif model_type == "vgg16":
            self.plain_model = build_vgg16(input_shape, label_count)
        else:
            raise ValueError(
                f"The model abbreviation '{model_type}' in the config file is wrong. Please fix this and try again."
            )

        self.plain_model.summary()
        # init the model
        self.model = DnnModel(model=self.plain_model)

        print(f"Using eta {self.config.data['learning_rate']}")
        optimizer = SGD(
            lr=self.config.data["learning_rate"]
        )

        # compile model
        self.model.compile(optimizer=optimizer,
                           loss=self.config.data["loss"],
                           metrics=self.config.data["metrics"])

    def setup_clients(self):
        # initialize the clients
        client_count = self.config.data["client_count"]
        for client_counter in np.arange(0, client_count):
            client = FederatedClient(
                id=client_counter,
                model=self.model,
                train_data=self.mnist_train_data_splits[client_counter],
                config=self.config
            )
            self.clients += [client]

    def setup_server(self):
        # initialize the server
        self.server = FederatedServer(
            global_gradients=self.plain_model.get_weights(),
            config=self.config
        )

    def simulate_training(self):
        # federated training
        import datetime

        max_iterations: int = self.config.data["max_iterations"]
        print(f"Federated training with {self.client_count} clients started")
        for iteration_counter in np.arange(1, max_iterations + 1):
            start = datetime.datetime.now()
            print(f'{"-" * 40}\nIteration\t{iteration_counter}\n{"-" * 40}')
            # each client
            for index, client in enumerate(self.clients):
                # get the global gradients from the server
                client.obtain_global_parameters(self.server)
                # train local model
                client.train_model()
                # send local gradients to server
                client.send_local_parameters(self.server)

            self.model.set_weights(self.server.global_gradients)
            x_test, y_test = self.dataset.test_data

            i_accuracy, i_loss = self.model.test(x_test, y_test)
            print(f"Accuracy: {i_accuracy} \n Loss: {i_loss}")
            grads = self.server.total_gradients

            self.logger.add_metrics(iteration=iteration_counter, accuracy=i_accuracy,
                                    loss=i_loss, grads=grads)
            elapsed_time = datetime.datetime.now() - start
            total_time = elapsed_time * (max_iterations - iteration_counter)
            print(f"Time left {total_time} - {elapsed_time} for this iteration")

        self.logger.export_data()

        # test the model
        print("Testing the final global model ...")
        self.model.set_weights(self.server.global_gradients)
        x_test, y_test = self.dataset.test_data
        self.model.test(x_test, y_test)

    def run(self):
        self.process_data()
        self.setup_model()
        self.setup_server()
        self.setup_clients()
        self.simulate_training()
