import pytest
from utils.model_builds import (build_two_layer_percepron,
                                build_small_cnn_model,
                                build_tl_cnn_model)


INPUT_SHAPE_GREY = (28, 28, 1)
INPUT_SHAPE_COLOR = (32, 32, 3)
LABEL_COUNT = 10

@pytest.mark.parametrize("shape", [INPUT_SHAPE_GREY, INPUT_SHAPE_COLOR])
def test_build_all(shape):
    build_tl_cnn_model.build_tl_cnn_model(shape, LABEL_COUNT)
    build_small_cnn_model.build_small_cnn_model(shape, LABEL_COUNT)
    build_tl_cnn_model.build_tl_cnn_model(shape, LABEL_COUNT)

