"""
    Copyright (C) 2019 Adrian Edward Thomas Henkel

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

from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPooling2D, Flatten
from keras.models import Sequential


def build_tl_cnn_model(input_shape, label_count):
    print("Building three layered cnn model ...")

    model = Sequential()

    # first hidden layer
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), input_shape=input_shape)
    )
    model.add(Activation("relu"))

    # second hidden layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # third hidden layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3)))
    model.add(Activation("relu"))

    # fourth hidden layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # fifth hidden layer
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation("relu"))

    # sixth hidden layer
    model.add(Flatten())

    # seventh hidden layer
    model.add(Dense(1024))
    model.add(Activation("relu"))

    # output layer
    model.add(Dense(label_count))
    model.add(Activation("softmax"))

    return model
