# first neural network
# TensorFlow Keras supervised learning 
# based on the tutorial from Jason Brownlee
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import os.path

#
# NN field defaults
#

default_n_1 = 12
default_n_2 = 16
default_n_3 = 1

default_activation_1 = 'relu'
default_activation_2 = 'relu'
default_activation_3 = 'sigmoid'

default_epoch_count = 150
default_batch_size = 10

#
# Flags
#
load_data = False
is_loaded_data = False
draw_learning_curve = False
# 0 is false, 1 is semi verbose, 2 is very verbose
verbose = 1

#
# Constants
#
test_values_filename = "test_values.csv"

with open(test_values_filename) as test_values_file:
    test_values = np.loadtxt(test_values_file, delimiter=",")


def nn(n_1=default_n_1, n_2=default_n_2, n_3=default_n_3,
       epoch_count=default_epoch_count, batch_size=default_batch_size,
       activation_1=default_activation_1, activation_2=default_activation_2, activation_3=default_activation_3):

    # load dataset
    dataset = np.loadtxt('pima-indians-diabetes.data.csv', delimiter=',')

    # load weights if they exist
    if os.path.isfile("model.json") and load_data:
        json_file = open("model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("model.h5")
        loaded_data = True
        print("Loaded model from disk")

    # input variables
    in_vars = dataset[:, 0:8]
    out_vars = dataset[:, 8]

    if not is_loaded_data:
        model = Sequential()
        model.add(Dense(n_1, input_shape=(8,), activation=activation_1))
        model.add(Dense(n_2, activation=activation_2))
        model.add(Dense(n_3, activation=activation_3))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if is_loaded_data:
        print("** Loaded Data: ")
        accuracy = model.evaluate(in_vars, out_vars)[1]
        print("%s: %.2f%%" % (model.metrics_names[1], accuracy * 100))

    print(model.summary())
    # input("** Begin training?: %d epochs | %d batch size" % (epoch_count, batch_size))
    # go over the data 150 times, updating the weights every 10 rows
    model.fit(in_vars, out_vars, epochs=epoch_count, batch_size=batch_size, verbose=verbose)

    # evaluate keras model
    accuracy = model.evaluate(in_vars, out_vars)[1]
    print("%s: %.2f%%" % (model.metrics_names[1], accuracy * 100))

    with open("history.csv", "a") as history_file:
        history_file.write(
            "\n%d, %d, %d, %d, %d, %.2f" % (n_1, n_2, n_3, epoch_count, batch_size, accuracy * 100))

    #
    # Save model
    #
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")
    print("Saved model to disk")

    #
    # Generate Learning Curve
    #
    if draw_learning_curve:
        print("generating learning curve plot")
        train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), in_vars, out_vars, cv=10,
                                                                scoring='accuracy', n_jobs=-1,
                                                                train_sizes=np.linspace(0.01, 1.0, 50))

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        print("launching learning curve plot")
        plt.subplots(1, figsize=(10, 10))
        plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
        plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

        plt.title("Learning Curve")
        plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


def main():
    nn()


main()
