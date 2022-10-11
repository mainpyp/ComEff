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

    Author: Adrian Edward Thomas Henkel, Reza NasiriGerdeh

"""

import numpy as np
import time

from utils.efficiency_utils import quantify_weights
from utils.efficiency_utils import boolean_to_integer
from utils.efficiency_utils import integer_to_boolean
from utils.config_reader import ConfigReader

import tensorflow as tf


class FederatedClient:
    def __init__(self, id: int, model, train_data, config: ConfigReader):
        print(f"Initializing client {id}...")
        self.id = id
        self.model = model
        self.train_data = train_data
        self.batch_size = config.data["batch_size"]
        self.learning_rate = config.data["learning_rate"]
        self.delta_gradients = []

        # efficiency params
        self.quantification_flag = config.data["quantification_flag"]
        self.quantification_options = config.data["quantification_options"]
        self.more_local_updates_flag = config.data["more_local_updates_flag"]
        self.more_local_updates_options = config.data["more_local_updates_options"]
        self.sparsification_flag = config.data["sparsification_flag"]
        self.sparsification_options = config.data["sparsification_options"]
        self.sparse_integer_array = None
        self.sparse_float_array = None
        if self.sparsification_flag:
            self.sparse_integer_array = []
            self.sparse_float_array = []

    def train_model(self):
        x_batch_list, y_batch_list = self.process_batches()

        local_loss, local_accuracy = None, None
        old_grads = self.model.get_weights()
        for x_batch, y_batch in zip(x_batch_list, y_batch_list):
            local_loss, local_accuracy = self.model.train(x_batch, y_batch)

        if self.more_local_updates_flag:
            max_local_iterations = self.more_local_updates_options["local_iterations"]
            for current_local_iteration in np.arange(1, max_local_iterations):
                print(f"Local epoch number: {current_local_iteration}")
                x_batch_list, y_batch_list = self.process_batches()
                for x_batch, y_batch in zip(x_batch_list, y_batch_list):
                    local_loss, local_accuracy = self.model.train(x_batch, y_batch)

        updated_grads = self.model.get_weights()
        self.delta_gradients = (
            np.array(old_grads) - np.array(updated_grads)
        ) / self.learning_rate
        return local_loss

    def send_local_parameters(self, server):
        integer_array = None

        local_gradient_updates = self.delta_gradients
        if self.quantification_flag:
            local_gradient_updates = quantify_weights(
                self.delta_gradients, self.quantification_options["dtype"]
            )

        if self.sparsification_flag:
            self.sparsify()
            local_gradient_updates = self.sparse_float_array

        server.receive_local_gradients(
            local_gradient_updates,
            len(self.train_data),
            integer_array=self.sparse_integer_array,
        )

    def obtain_global_parameters(self, server):
        print(f"Client {self.id} receiving global model.")
        global_weights = server.get_global_gradients()

        if self.quantification_flag:
            global_weights = quantify_weights(global_weights, np.float32)

        self.model.set_weights(global_weights)

    def process_batches(self):
        np.random.shuffle(self.train_data)
        x_shuffled, y_shuffled = zip(*self.train_data)

        batch_count = np.ceil(len(x_shuffled) / self.batch_size)

        x_batch_list = np.array_split(x_shuffled, batch_count)
        y_batch_list = np.array_split(y_shuffled, batch_count)

        return x_batch_list, y_batch_list

    def sparsify(self):
        """
        This method first calculates the delta matrix between the weights of the previous training and the current
        training. Then the percentile of each layer is calculated and all elements under this percentile are removed.
        The remaining values are stored in sparse_float_array and the positions of the changed weights are stored in
        sparsification_boolean_array.
        :return: None
        """
        print("Start Sparsification on client side:")
        self.sparse_integer_array = []
        self.sparse_float_array = []

        delta_array = self.delta_gradients
        for delta_layer in delta_array:
            # get the percentile of the whole layer
            percentile = np.percentile(
                np.abs(delta_layer), self.sparsification_options["percentile"]
            )
            #  sets all values over the percentile to true, others to false
            boolean_layer = np.abs(delta_layer) >= percentile
            # deletes all values that are not true in the boolean array
            sparse_layer = delta_layer[boolean_layer]

            # transforms every boolean array in each layer into a bit array
            integer_array = boolean_to_integer(boolean_layer.flatten())
            self.sparse_integer_array.append(integer_array)
            self.sparse_float_array.append(sparse_layer)
        print("End sparsification on client side")
