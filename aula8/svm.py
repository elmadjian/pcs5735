import sys
import numpy as np
from sklearn import svm
from sklearn import metrics

filename = sys.argv[1]
dataset  = []
print("processing input file...")
with open(filename, "r") as f:
    line = f.readline()
    while line:
        line = f.readline()
        strs = line.split(',')
        if len(strs) == 1:
            break
        vec  = [int(i) for i in strs]
        dataset.append(vec)
    dataset = np.array(dataset)

print("training the classifier...")
X = dataset[:, :-1]
Y = dataset[:, -1]

split = 0.75

spltv = int(split * len(Y))
X_train = X[:spltv,:]
Y_train = Y[:spltv]
X_test  = X[spltv:,:]
Y_test  = Y[spltv:]

svc = svm.SVC(kernel='poly', degree=2)
svc.fit(X_train, Y_train)

print("testing data...")
predicted = svc.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))
