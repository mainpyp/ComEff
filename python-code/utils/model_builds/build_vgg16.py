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
from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential
from keras import regularizers
import keras


def build_vgg16(input_shape: tuple, label_count: int):
    model = keras.applications.VGG16(
        include_top=True, weights=None, input_tensor=None,
        input_shape=input_shape, pooling=None, classes=label_count
    )
    return model
