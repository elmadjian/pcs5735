'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''

from __future__ import print_function
import numpy as np
import sys
np.random.seed(1337)  # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils

def main():
    if len(sys.argv) != 2:
        print("usage: <this_program> <dataset_file>")
        sys.exit()

    nb_epoch = 50
    print('Loading data...')
    dataset, target = load_dataset(sys.argv[1])
    training_set, training_target = dataset[:-25], target[:-25]
    testing_set, testing_target   = dataset[-25:], target[-25:]
    print(len(training_set), 'train sequences')
    print(len(testing_set), 'test sequences')

    nb_classes = 2
    print(nb_classes, 'classes')

    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    training_target = np_utils.to_categorical(training_target, nb_classes)
    testing_target  = np_utils.to_categorical(testing_target, nb_classes)
    # print('Y_train shape:', Y_train.shape)
    # print('Y_test shape:', Y_test.shape)

    print('Building model...')
    #print("shape:", len(training_set[0]))
    model = Sequential()
    model.add(Dense(24, input_dim=24))
    model.add(Activation('tanh'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(training_set, training_target,
                        nb_epoch=nb_epoch, batch_size=16,
                        verbose=1, validation_split=0.1)
    score = model.evaluate(testing_set, testing_target,
                       batch_size=16, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

#--------------------------
def load_dataset(filename):
    dataset, target = [], []
    with open(filename, 'r') as f:
        for line in f:
            values  = line.split()
            numeric = [float(v) for v in values]
            dataset.append(numeric[:-1])
            target.append(numeric[-1]-1)
            #target.append([1,0]) if numeric[-1] == 1.0 else target.append([0,1])
    return dataset, target

#-----------------------
if __name__=="__main__":
    main()
