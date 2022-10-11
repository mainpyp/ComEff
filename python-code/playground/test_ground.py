import numpy as np
import tensorflow as tf


def datetime_test():
    array = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    data = np.arange(60000, dtype=tf.Tensor)
    data.fill(array)

    import datetime

    start = datetime.datetime.now()

    split_data = np.array_split(data, len(data) / 65)

    end = datetime.datetime.now()
    delta_time = end - start


def test_percetile():
    test_arrays = np.array(
        [
            [1, 2, 3, 4, 5, 6, 100],
            [2, 3, 4, 5, 6, 6, 100],
            [1, 2, 3, 4, 5, 6, 100],
            [2, 3, 4, 5, 6, 6, 100],
        ],
        np.int32,
    )

    print(f"Percentile {np.percentile(test_arrays, 86)}")
    print(f"Cut {test_arrays < 6.0}")

    bool_array = np.array(
        [
            [True, False, True, True, False, False],
            [True, False, True, True, False, False],
            [True, False, True, True, False, False],
            [True, False, True, True, False, False],
        ],
        np.int32,
    )

    for i, array in enumerate(test_arrays):
        perc = np.floor(np.percentile(array, 50))
        boolean_array = np.where(array > perc, True, False)
        print(array[boolean_array])
        now = np.where(np.invert(boolean_array), array, -2)
        occ = np.count_nonzero(boolean_array == True)
        diff = boolean_array * now
        diff = diff[diff != -2]


def fill_zeros():
    whole = np.array([1, 0, 3, 0, 6])

    sparse = np.array([8, 9])

    whole[whole == 0] = sparse
    print(whole)


def test_zip():
    firsts = [1, 2, 3, 4, 5, 6]
    seconds = ["one", "two", "three", "four", "five", "six"]
    thirds = ["first", "second", "third", "fourth", "fifth", "sixth"]

    for first, second, third in zip(firsts, seconds, thirds):
        print(first, second, third)


def test_utils():
    import utils.efficiency_utils as uts

    one_test = np.array([True, True, True, True, True, True, True, True, True])
    converded = uts.boolean_to_integer(one_test)
    recalculated = uts.integer_to_boolean(converded, len(one_test))
    print(one_test)  # [ True  True  True  True  True  True  True  True  True  True]
    print(recalculated)  # [ True False False False False False False  True  True  True]


def test_where():
    val = np.array([1, 2, 3, 4, 5, 6])
    boo = np.array([False, True, False, True, False, True])
    insert = np.array([10, 11, 12])
    val[boo] = insert
    #moin = val[boo]
    #result = np.where(boo[val], insert)
    #print(moin)
    print(val)


def delta():
    first = np.array(
        [3, 3, 0, 3, 3]
    )
    second = np.array(
        [1, 1, 1, 1, 1]
    )

    deltaa = first - second
    deltaa = np.abs(deltaa)
    print(deltaa)

def parse():
    import argparse as args
    parser = args.ArgumentParser()
    parser.add_argument("--perc", "-percentile", type=int)
    args = parser.parse_args()

    return args.perc

def test_append():
    test_data = np.array([[2, 4]])
    append_data = np.array([[1, 2]])
    test_data = np.append(test_data, append_data, axis=0)
    print(test_data)

def test_append_pandas():
    from time import sleep
    import pandas as pd
    df = pd.DataFrame(columns=["Hund", "Katze", "Maus"])
    for i in range(14):
        sleep(3)
        print(f"Sleep round {i} done")
        df.loc[i] = ['name' + str(i)] + [f"Hund {i}", f"Katze {i}"]
        df.to_csv("test.csv", sep=";")




if __name__ == "__main__":
    test_append_pandas()