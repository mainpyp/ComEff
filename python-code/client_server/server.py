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

from utils.efficiency_utils import quantify_weights
from utils.efficiency_utils import integer_to_boolean
from utils.config_reader import ConfigReader


class FederatedServer:
    def __init__(self, global_gradients, config: ConfigReader):
        print("Initializing the server ...")
        self.global_gradients = np.array(global_gradients)
        self.old_gradient_updates = [
            np.zeros(layer.shape) for layer in self.global_gradients
        ]

        self.client_count = config.data["client_count"]
        self.learning_rate = config.data["learning_rate"]
        self.local_gradient_updates_from_clients = []
        self.sample_count_from_clients = []
        self.receive_counter = 0
        # efficiency params
        self.quantification_flag = config.data["quantification_flag"]
        self.quantification_options = config.data["quantification_options"]
        self.sparsification_flag = config.data["sparsification_flag"]

        self.total_gradients = 0

    def get_global_gradients(self):
        if self.quantification_flag:
            global_weights = quantify_weights(
                self.global_gradients, self.quantification_options["dtype"]
            )
        else:
            global_weights = self.global_gradients

        return global_weights

    def receive_local_gradients(
        self, local_gradients, local_sample_count, integer_array: list = None
    ):
        post_processed_local_gradients = local_gradients
        if self.quantification_flag:
            post_processed_local_gradients = quantify_weights(
                local_gradients, np.float32
            )

        if self.sparsification_flag:
            post_processed_local_gradients = self.desparsify(
                post_processed_local_gradients, integer_array
            )

        self.local_gradient_updates_from_clients += [post_processed_local_gradients]

        self.sample_count_from_clients += [local_sample_count]
        self.receive_counter += 1

        # if local gradients received from all clients, start aggregation and updating global gradients
        if self.receive_counter == self.client_count:
            average_gradient_updates = []
            layer_count = len(self.global_gradients)
            for layer_counter in range(0, layer_count):
                sum_gradient_updates = np.zeros(
                    self.global_gradients[layer_counter].shape
                )
                global_sample_count = 0
                for local_gradient_updates, local_sample_count in zip(
                    self.local_gradient_updates_from_clients,
                    self.sample_count_from_clients,
                ):
                    sum_gradient_updates = (
                        sum_gradient_updates
                        + local_gradient_updates[layer_counter] * local_sample_count
                    )
                    global_sample_count += local_sample_count

                average_gradient_updates += [
                    (sum_gradient_updates / global_sample_count)
                ]
            # Federated SGD
            self.global_gradients = (
                self.global_gradients
                - self.learning_rate * np.array(average_gradient_updates)
            )
            self.receive_counter = 0
            self.old_gradient_updates = np.array(average_gradient_updates)
            self.local_gradient_updates_from_clients = []
            self.sample_count_from_clients = []

    def desparsify(self, sparse_gradient_updates, integer_array):
        """
        This method takes the sparse gradients as well as the boolean from the client.
        It goes through every lists list, compares it with the inverted boolean list values.
        The inverted boolean value is taken because all False values are replaced with 2.0 which is later on replaced
        with the corresponding values of the array with the sparse values from the client.
        :param sparse_gradients: All gradients that were over a given percentile.
        :param integer_array: The integer array that was transformed by the utils script. Each value tells if
        :return: The array with the replaced weights.
        """
        print("Starts desparsifcation on server side")

        transformed = []

        for sparse_layer, integer_layer, old_layer in zip(
            sparse_gradient_updates, integer_array, self.old_gradient_updates
        ):
            boolean_layer = integer_to_boolean(integer_layer, len(old_layer.flatten()))
            transformed_layer = []
            sparse_grad_index = 0

            sparse_grad_count = len(sparse_layer)
            true_count = np.where(boolean_layer == True)[0].size

            assert (
                sparse_grad_count == true_count
            ), "Something is wrong with binary/integer or vice versa conversion"

            for old_update, new_update_flag in zip(old_layer.flatten(), boolean_layer):
                if new_update_flag:
                    new_update = sparse_layer[sparse_grad_index]
                    transformed_layer += [new_update]
                    sparse_grad_index += 1

                    self.total_gradients += 1
                else:
                    transformed_layer += [old_update]

            transformed_layer = np.array(transformed_layer).reshape(old_layer.shape)
            transformed.append(transformed_layer)
        print("Ends desparsifcation on server side")
        return transformed
