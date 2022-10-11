"""
    Copyright (C) 2020 Adrian Edward Thomas Henkel, Reza NasiriGerdeh

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
from typing import Union


def integer_to_boolean(integer_array, gradient_count):
    boolean_array = []
    for integer in integer_array:
        binary_string = "{0:08b}".format(integer)
        for char in binary_string:
            if char == "0":
                boolean_array.append(False)
            else:
                boolean_array.append(True)

    if gradient_count % 8 != 0:
        padded_bits = len(integer_array) * 8 - gradient_count
        last_byte = boolean_array[padded_bits - 8 :]
        boolean_array = np.concatenate((boolean_array[0:-8], last_byte))

    return np.array(boolean_array)


def boolean_to_integer(boolean_array):
    zero_one_array = np.where(boolean_array == True, 1, boolean_array)
    zero_one_array = np.where(zero_one_array == False, 0, zero_one_array)

    array_size = len(zero_one_array)

    if array_size % 8 != 0:
        padding_zeros = np.zeros(8 - (array_size % 8))
        last_bits = zero_one_array[-(array_size % 8) :]
        padded_last_byte = np.concatenate((padding_zeros, last_bits))

        zero_one_array = np.concatenate(
            (zero_one_array[0 : -(array_size % 8)], padded_last_byte)
        ).astype(np.uint8)

    integer_array = []
    for eight_bits in np.split(zero_one_array, len(zero_one_array) // 8):
        integer = int("".join([str(bit) for bit in eight_bits]), 2)
        integer_array.append(integer)
    result = np.array(integer_array).astype(np.uint8)
    import sys
    #  print(f"Before: {sys.getsizeof(boolean_array)}")
    #  print(f"After: {sys.getsizeof(result)}")
    return np.array(integer_array).astype(np.uint8)


def quantify_weights(
    layers: list, transform_type: str
) -> list:
    """This method transforms each weights and biases of np.float type into a given type.
    :param layers: list with np.ndarrays, each symbolizing one nn layer
    :param transform_type: np.float16, np.float32, np.float64
    :return: The transformed weights
    """
    if transform_type == "float16":
        transformed_list = [np.float16(layer) for layer in layers]
    elif transform_type == "float32":
        transformed_list = [np.floa32(layer) for layer in layers]
    elif transform_type == "float64":
        transformed_list = [np.floa64(layer) for layer in layers]
    else:
        transformed_list = layers
    #transformed_list = [layer.astype(transform_type) for layer in layers]
    return transformed_list
