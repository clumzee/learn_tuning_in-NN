import os
import warnings
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow.keras
import mlflow
import logging

from tensorflow.keras.datasets import mnist


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    #from line 24 to 30 we are getting input of all the neccesary parameters to make our model from the user using command line code

    mlflow.keras.autolog()
    with mlflow.start_run():

        no_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        no_neurons = int(sys.argv[2]) if len(sys.argv) > 2 else 128
        activation_func = sys.argv[3] if len(sys.argv) > 3 else "relu"
        optimizer_func = sys.argv[4] if len(sys.argv) > 4 else "adam"
        epochs_num = int(sys.argv[5]) if len(sys.argv) > 5 else 5
        loss_function = sys.argv[6] if len(sys.argv) > 6 else "sparse_categorical_crossentropy"
        batch_norm = int(sys.argv[7]) if len(sys.argv) > 7 else 0


        #34th line loads the mnist data (As per the given challenge)
        (x_train_mnist,y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

        print(x_train_mnist.shape)

        print(no_layers, no_neurons,activation_func, optimizer_func, epochs_num, loss_function, batch_norm)


        # now we are creating the list of layers as per the first argument of the command
        #layers list already has one layer which is to flatten the mnist image
        layers_model = [tf.keras.layers.Flatten(input_shape = (28,28,1))]

        #here if user put 1 for batchnormalization(in the command) then the below code block will add a batchnormalization layer after every
        #three dense layers otherwise only dense layers will be created 
        if batch_norm == 0:
            for i in range(no_layers):
                layers_model.append(tf.keras.layers.Dense(no_neurons, activation=activation_func))
        else:
            for i in range(no_layers):
                if i %3 == 0:
                    layers_model.append(tf.keras.layers.Dense(no_neurons, activation=activation_func))
                    layers_model.append(tf.keras.layers.BatchNormalization())
                else:
                    layers_model.append(tf.keras.layers.Dense(no_neurons, activation=activation_func))

        layers_model.append(tf.keras.layers.Dense(10, activation = "softmax"))

        print(layers_model)

        #Now making the model using the layers list created above
        mnist_model = tf.keras.Sequential(layers_model)

        #Compiling the model using the optimizer_func given by the user in command line only
        mnist_model.compile(optimizer_func, loss = loss_function, metrics=["accuracy"])

        #training the model 
        history = mnist_model.fit( x_train_mnist  ,  y_train_mnist, epochs=epochs_num)

        evaluation = mnist_model.evaluate(x_test_mnist, y_test_mnist)