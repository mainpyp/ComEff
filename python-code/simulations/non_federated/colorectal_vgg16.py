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
import sys

sys.path.insert(0, "../..")

from benchmarking.logger import Logger
from utils.config_reader import ConfigReader
from utils.dataset import Dataset
from utils.dataset_type import DatasetType
from utils.model_builds.build_vgg16 import build_vgg16
from utils.model import DnnModel

from keras.optimizers import SGD


# simulation parameters
dataset_name = "colorectal"
dataset_path = "../datasets/colorectal"
dataset_type = DatasetType.COLOR_IMAGE
image_shape = (150, 150)
data_type = "float32"
max_value = 255
label_count = 8
batch_size = 64
learning_rate = 0.0025
loss = "categorical_crossentropy"
metrics = ["accuracy"]
max_iterations = 300


config = ConfigReader("../../configs/final_run/colorectal/vgg16/non_federated.yaml")
logger = Logger(config)

logger.set_filename("non_federated")

# open and prepare mnist dataset
dataset = Dataset(
    config=config
)

# open mnist dataset
dataset.open()

# pre-process the dataset
dataset.preprocess(max_value=max_value, label_count=label_count)

# build the model
input_shape = dataset.train_data[0][0].shape
label_count = dataset.train_data[0][1].shape[0]
model = build_vgg16(input_shape, label_count)

# compile the model
mnist_small_model = DnnModel(model=model)
optimizer = SGD(lr=learning_rate)

mnist_small_model.model.summary()

mnist_small_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# train the model
import datetime

print("Training the vgg16 model ...")
for iteration_counter in range(1, max_iterations + 1):
    start = datetime.datetime.now()
    print(f"iteration {iteration_counter}")
    x_batch_list, y_batch_list = dataset.get_batch(batch_size)

    for batch_index, (x_batch, y_batch) in enumerate(zip(x_batch_list, y_batch_list)):
        mnist_small_model.train(x_batch, y_batch)

    print("Testing the vgg16 model ...")
    x_test, y_test = dataset.test_data
    i_accuracy, i_loss = mnist_small_model.test(x_test, y_test)

    logger.add_metrics(iteration_counter, accuracy=i_accuracy, loss=i_loss)
    print(f"Loss: {i_loss}")
    print(f"Accuracy: {i_accuracy}")

    elapsed_time = datetime.datetime.now() - start
    full_time = elapsed_time * (max_iterations - iteration_counter)
    print(f"Expected time left {full_time} - {elapsed_time} for this iteration")
    print()

# test the model
print("Testing the vgg16 model ...")
x_test, y_test = dataset.test_data
mnist_small_model.test(x_test, y_test)

logger.export_data()
