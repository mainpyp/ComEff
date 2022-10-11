"""
    Copyright (C) 2020 Adrian Edward Thomas Henkel, Reza Naserigerdeh

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

    Author: Adrian Edward Thomas Henkel, Reza Naserigerdeh

"""

from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Sequential


def build_two_layer_perceptron(input_shape: tuple, label_count: int) -> Sequential:
    """Instantiates a two layer perceptron with 2 hidden dense layers, each containing
    200 neurons.

    Parameters:
    -----------
    input_shape:
        Shape of the numpy array representing the image
    label_count:
        Number of different labels (MNIST = 0-9 = 10

    Returns:
    --------
        model: Sequential
    """
    print("Building the two layer perceptron model ...")

    model = Sequential()

    model.add(Flatten(input_shape=input_shape))

    # first hidden layer
    model.add(Dense(units=200))
    model.add(Activation("relu"))

    # second hidden layer
    model.add(Dense(units=200))
    model.add(Activation("relu"))

    # output layer
    model.add(Dense(units=label_count, activation="softmax"))

    return model
