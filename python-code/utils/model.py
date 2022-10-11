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

import os
import numpy as np
import tensorflow as tf


class DnnModel:
    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss, metrics):
        self.model.compile(
            optimizer=optimizer, loss=loss, metrics=metrics,
        )

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def train(self, x_batch, y_batch):
        if y_batch.shape == (64, 1, 10):
            reshaped = []
            for each in y_batch:
                reshaped.append(each[0])
            reshaped = np.array(reshaped)
            assert reshaped.shape == (64, 10), "Error, the shape is not correct!"
            [loss, accuracy] = self.model.train_on_batch(x=x_batch, y=reshaped)
        else:
            [loss, accuracy] = self.model.train_on_batch(x=x_batch, y=y_batch)
        return loss, accuracy

    def test(self, x_test, y_test):
        # To convert y test into one hot encoded
        y = np.zeros((y_test.size, y_test.max() + 1))
        if isinstance(y_test[0], np.uint8):  # if y label is format [1, 2, 5, 6]
            for index, label in enumerate(y):
                label[y_test[index]] = 1
        elif len(y_test[0]) == 1:  # if y label has format [[1], [2]]
            for index, label in enumerate(y):
                label[y_test[index][0]] = 1
        else:
            y[np.arange(y_test.size), y_test] = 1
        loss, accuracy = self.model.evaluate(x_test, y)

        return accuracy, loss

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        weights = self.model.get_weights()
        for layer_counter in range(len(weights)):
            layer_weights = weights[layer_counter]
            np.save(f"{save_dir}/layer_{layer_counter}_weights", layer_weights)

    def restore(self, restore_dir):
        if not os.path.exists(restore_dir):
            print(f"{restore_dir} not found!")
            return

        weights = []
        for layer_counter in range(len(os.listdir(restore_dir))):
            layer_weights = np.load(f"{restore_dir}/layer_{layer_counter}_weights.npy")
            weights += [layer_weights]

        self.model.set_weights(weights=weights)
