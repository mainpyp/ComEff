"""
This script is generated following and slightly changing this tutorial:
https://www.youtube.com/playlist?list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU

List of activation functions: https://www.tensorflow.org/api_docs/python/tf/keras/activations
    relu: f(x) = max(0, x)
    sigmoid: sig(x) = 1 / 1 + e^-t = e^x / 1 + e^x  ~  from ca. -6 to 6
    softmax: softmax-i(a) = e^ai / sum-k(e^ak)
        i -> input index
        a -> input array
        e^ai is e to the power of a[i]
        sum-k(e^ak) is the sum of all e^ai
        e.g.
            a = [1, 2, 3, 4, 5, 6]
            e^a = [2.72, 7.39, 20.08, 54.60, 148.41, 403.43]
            sum-k(e^ak) = sum([2.72, 7.39, 20.08, 54.60, 148.41, 403.43]) = 636.63
            so: softmax-0(a) = 2.72 / 636.63 -> 0.004
                softmax-5(a) = 403.43 / 636.63 -> 0.634
        p of all elements of a add up to 1.
List of all Layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from nn_helpers import visualize_matrix

"""
Summary:
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 32)                352       
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 66        
=================================================================
Total params: 418
Trainable params: 418
Non-trainable params: 0
_________________________________________________________________
"""
my_model = Sequential(
    [
        Dense(
            16, input_shape=(1,), activation="softmax"
        ),  # first hidden layer, has to implement input_shape!!!
        Dense(32, activation="relu"),
        Dense(128, activation="softmax"),
        Dense(128, activation="softmax"),
        Dense(2, activation="softmax"),  # output layer
    ]
)

my_model.summary()

"""
epoch: one run through the model during training
gradient: other term for multivariable derivation 

gradient(loss)
--------------    +   learning_rate 
gradient(weight) 

compile:
    Optimizer:  https://keras.io/optimizers/
                Adam is a variant of stochastic gradient descent
                learning rate is a float normally between 0.0001 and 0.1 
                When lr is to high, chances are high that you overshoot over the minimum
    Loss:       https://keras.io/losses/
                Loss function that calculates the performance of each epoch
                
    Metrics:    https://keras.io/metrics/
                Accuracy T / n
fit:  
    train: np.array value for each element
    train labeled: np.array labels for each element in train
    batch_size: how many pieces of data we want to give the model at once
    epochs: how many times the model is updated with the batch size elements
    shuffle: optional 
    verbose: optional 
             0 silent
             1 progress bar
             2 one line per epoch
             
train set: all data that a model sees during training and is split up into batches ~0.7
validation set: the model never sees the data but is used during for testing if the model is over fitting oder not ~0.15
test set: not labeled set of data that is used for evaluate the final model ~0.15

to specify the validation set:
set = [(sample1, label1), (sample2, label2), (sample3, label3)
fit(..., validation_data = set, ...) 
"""

from nn_data import get_simulated_data
from tensorflow.keras.optimizers import Adam

my_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

train_samples, train_labels = get_simulated_data(scaled=False)

my_model.fit(
    train_samples,
    train_labels,
    validation_split=0.15,
    batch_size=25,
    epochs=15,
    shuffle=True,
    verbose=1,
)

# to predict you only have to pass in a np.array or a list(like) object of elements you want to predict
predictions = my_model.predict([15, 20, 24, 12, 100, 99, 88, 65, 64, 63, 66], verbose=2)
print(predictions)
"""
[[0.87612075 0.12387925]
 [0.87610817 0.12389181]
 [0.8760948  0.12390515]
 [0.87612647 0.12387351]
 [0.12777951 0.87222046]
 [0.12778242 0.87221766]
 [0.12789577 0.87210417]
 [0.6332421  0.3667579 ]
 [0.74621874 0.2537813 ]
 [0.80776894 0.19223107]
 [0.47486386 0.5251362 ]]
"""


if __name__ == "__main__":
    pass
