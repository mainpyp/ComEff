import numpy as np


float_ndarray = np.array(
    [
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    ]
)

type_before_parsing = type(float_ndarray[0][0][0])  # <class 'numpy.float64'>


float_ndarray = float_ndarray.astype(np.float16)

type_after_parsing = type(float_ndarray[0][0][0])

if __name__ == "__main__":
    print(f"Type before parsing : {type_before_parsing}")
    print(f"Type after parsing : {type_after_parsing}")
    print(f"type comprehension : {type([x for x in range(91)])}")
    print(f"type parsed comprehension : {type(list([x for x in range(91)]))}")
