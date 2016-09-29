#Author: Carlos Eduardo Leão Elmadjian
#---------
#Be aware of dependencies... (Numpy, Matplotlib, Keras, Theano...)

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import Callback

#Necessary to recover accuracy data from tests
#---------------------------------------------
class TestCallback(Callback):
    def __init__(self, testing_set, testing_target):
        self.testing_set = testing_set
        self.testing_target = testing_target
        self.accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        x = self.testing_set
        y = self.testing_target
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.accuracy.append(acc)

#initial setup
#--------------
def main():
    if len(sys.argv) != 2:
        print("usage: <this_program> <dataset_file>")
        sys.exit()

    #initial settings
    np.random.seed(7)
    nb_epoch = 100

    #loading data
    print('Loading data...')
    dataset, target = load_dataset(sys.argv[1])
    training_set, training_target = dataset[:-30], target[:-30]
    testing_set, testing_target   = dataset[-30:], target[-30:]
    print(len(training_set), 'train sequences')
    print(len(testing_set), 'test sequences')
    nb_classes = 4
    print(nb_classes, 'classes')

    #normalizing classes
    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    training_target = np_utils.to_categorical(training_target, nb_classes)
    testing_target  = np_utils.to_categorical(testing_target, nb_classes)

    #building the network
    print('Building model...')
    model = Sequential()
    model.add(Dense(960, input_dim=960))
    model.add(Activation('tanh'))
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #training the network
    tester  = TestCallback(testing_set, testing_target)
    history = model.fit(training_set, training_target,
                        nb_epoch=nb_epoch, batch_size=534,
                        callbacks=[tester],
                        verbose=1, validation_split=0.1)

    #testing the network
    score = model.evaluate(testing_set, testing_target,
                       batch_size=30, verbose=1)

    #show the results
    print('\n\nTest score:', score[0])
    print('Test accuracy:', score[1])

    p1, = plt.plot(history.history['loss'], 'r-')
    p2, = plt.plot(history.history['acc'], 'b-')
    p3, = plt.plot(tester.accuracy, 'g-')
    plt.legend([p1, p2, p3], ['perda', 'acurácia', 'dados de teste'])
    plt.show()

#expects the file faces.csv as parameter
#---------------------------------------
def load_dataset(filename):
    """
    left     --> 0
    right    --> 1
    up:      --> 2
    straight --> 3
    """
    dataset, target = [], []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            values = line.split(',')
            direction = values[-1]
            numeric   = [float(i) for i in values[:-1]]
            dataset.append(numeric)
            if "left" in direction:
                target.append(0)
            elif "right" in direction:
                target.append(1)
            elif "up" in direction:
                target.append(2)
            elif "straight" in direction:
                target.append(3)
    return dataset, target


#-----------------------
if __name__=="__main__":
    main()
