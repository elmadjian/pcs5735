from __future__ import print_function
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import Callback

#------------------------
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


#------------------------
def main():
    #initial settings
    np.random.seed(7)
    nb_epoch = 100

    #loading data
    print('Loading data...')
    training_set, training_target = load_dataset('/faces_4')
    testing_set, testing_target   = load_dataset('/faces_test')
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

#------------------------
def load_dataset(folder_path):
    """
    left     --> 0
    right    --> 1
    up:      --> 2
    straight --> 3
    """
    dataset, target = [], []
    for root, dirs, files in os.walk(os.getcwd() + folder_path):
        for name in files:
            img = cv2.imread(os.path.join(root, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = img.flatten().tolist()
            dataset.append(features)
            if "_left_" in name:
                target.append(0)
            elif "_right_" in name:
                target.append(1)
            elif "_up_" in name:
                target.append(2)
            elif "_straight_" in name:
                target.append(3)
    return dataset, target


#-----------------------
if __name__=="__main__":
    main()
