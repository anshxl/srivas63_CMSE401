#Load MNIST data
import tensorflow.keras.datasets # type: ignore

def load_data():
    mnist = tensorflow.keras.datasets.mnist.load_data(
        path="mnist.npz"
    )

    #Split data into training and testing sets
    (x_train, y_train), (x_test, y_test) = mnist

    #Normalize pixel values to between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    #One-hot encode the labels
    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)