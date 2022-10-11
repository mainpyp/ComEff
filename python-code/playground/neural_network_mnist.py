from nn_data import get_mnist_data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D
from playground.nn_data import get_mnist_data


# explanation of cnns:
# https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
mnist_model = Sequential(
    [
        tf.keras.layers.Dense(200, input_shape=(28, 28, 1), activation="relu"),
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)


mnist_model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

mnist_model.summary()

sample_train, label_train, sample_test, label_test = get_mnist_data()
print("----- done getting data ----")
print(len(sample_train), len(label_train))

data = list(zip(sample_train, label_train))

# mnist_model.fit(sample_train, sample_test, epochs=15, verbose=2)
mnist_model.fit(sample_train, sample_test, epochs=15, verbose=2)

mnist_model.evaluate(sample_test, label_test, verbose=2)
