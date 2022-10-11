from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
from ann_visualizer.visualize import ann_viz

model = Sequential(
    [Dense(input_shape=(28, 28), units=200), Dense(units=200), Dense(units=10)]
)

model.compile()

ann_viz(model, filename="two_layer_perceptron.gv")


def test_function(param: str):
    """THis is the documentation of the test function

    Parameters
    ----------
    param : str
        This is a single parameter

    Returns
    -------
    None
        It does not return a single thing
    """
    return None


