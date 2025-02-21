#Load MNIST data
import tensorflow.keras.datasets # type: ignore
import numpy as np

def one_hot_encode(y):
    one_hot = np.zeros((y.size, y.max() + 1))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def transform_data(x):
    x = x.reshape(x.shape[0], -1)
    return x

def load_data():
    mnist = tensorflow.keras.datasets.mnist.load_data(
        path="mnist.npz"
    )

    #Split data into training and testing sets
    (x_train, y_train), (x_test, y_test) = mnist

    #Normalize pixel values to between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    #Transform data
    x_train = transform_data(x_train)
    x_test = transform_data(x_test)

    #One-hot encode the labels using NumPy
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return (x_train, y_train), (x_test, y_test)