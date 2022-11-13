import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# LOAD DATASET
inputs = np.load('inputs.npy')
outputs = np.load('labels.npy').flatten()

# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()


def getInputColor(label):
    if label == 1:
        return "blue"
    return "red"


def plotData():
    x = np.arange(0, 60, 1)
    y = np.arange(0, 60, 1)
    plt.xticks(x)
    plt.yticks(y)
    for i in range(200):
        for j in range(10):
            plt.scatter(j, inputs[i][j], c=getInputColor(outputs[i]))
    plt.show()


# plotData()

DATASET_SIZE = len(inputs)
TRAIN_SIZE = int(DATASET_SIZE * 0.75)
TEST_SIZE = int(DATASET_SIZE * 0.25)


print('INPUTS SHAPE', np.shape(inputs))
print('LABELS SHAPE', np.shape(outputs))

print('DATASET SIZE', DATASET_SIZE)
print('TRAIN SIZE', TRAIN_SIZE)
print('TEST SIZE', TEST_SIZE)

# FOR ALGO 1


def getUsefullData(inputs):
    return [inputs[0],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[7],
            inputs[8],
            inputs[10],
            inputs[13],
            inputs[17],
            inputs[23],
            inputs[24],
            inputs[26],
            inputs[28],
            inputs[32],
            inputs[36],
            inputs[40],
            inputs[41],
            inputs[45],
            inputs[49],
            inputs[52],
            inputs[53],
            inputs[54],
            inputs[55],
            inputs[56],
            inputs[59]]


train_data = [getUsefullData(i) for i in inputs[:TRAIN_SIZE]]
train_labels = outputs[: TRAIN_SIZE]

print()
print('TRAIN SHAPE', np.shape(train_data))
test_data = [getUsefullData(i) for i in inputs[TRAIN_SIZE:]]
test_labels = outputs[TRAIN_SIZE:]

# MODEL DEFINITION
logreg_clf = LogisticRegression()
m2 = KNeighborsClassifier()
m3 = DecisionTreeClassifier()
m4 = GaussianNB()
# MODEL TRAINING
print('MODEL TRAINING...')
logreg_clf.fit(train_data, train_labels)
m2.fit(train_data, train_labels)
m3.fit(train_data, train_labels)
m4.fit(train_data, train_labels)

# MODEL TESTING
print()
print('TEST SHAPE', np.shape(test_data))
print('MODEL TESTING...')
res = logreg_clf.predict(test_data)
res2 = m2.predict(test_data)
res3 = m3.predict(test_data)
res4 = m4.predict(test_data)

print()
print('MODEL MEAN ACCURACY', accuracy_score(test_labels, res) * 100, '%')
print('MODEL2 MEAN ACCURACY', accuracy_score(test_labels, res2) * 100, '%')
print('MODEL3 MEAN ACCURACY', accuracy_score(test_labels, res3) * 100, '%')
print('MODEL4 MEAN ACCURACY', accuracy_score(test_labels, res4) * 100, '%')
