import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# LOAD DATASET
inputs = np.load('inputs.npy')
# outputs = np.load('labels.npy')
outputs = [int(x[0]) for x in np.load('labels.npy')]

print(inputs)
print(outputs)
DATASET_SIZE = len(inputs)
TRAIN_SIZE = int(DATASET_SIZE * 0.75)
TEST_SIZE = int(DATASET_SIZE * 0.25)

print('INPUTS SHAPE', np.shape(inputs))
print('LABELS SHAPE', np.shape(outputs))

print('DATASET SIZE', DATASET_SIZE)
print('TRAIN SIZE', TRAIN_SIZE)
print('TEST SIZE', TEST_SIZE)

train_data = inputs[:TRAIN_SIZE]
train_labels = outputs[:TRAIN_SIZE]

print()
print('TRAIN SHAPE', np.shape(train_data))
test_data = inputs[TRAIN_SIZE:]
test_labels = outputs[TRAIN_SIZE:]

# MODEL DEFINITION
logreg_clf = LogisticRegression()
m2 = KNeighborsClassifier()
m3 = LinearDiscriminantAnalysis()
m4 = GaussianNB()
m5 = DecisionTreeClassifier()
m6 = SVC()
# MODEL TRAINING
print('MODEL TRAINING...')
# logreg_clf.fit(train_data, train_labels)
m2.fit(train_data, train_labels)
# m3.fit(train_data, train_labels)
m4.fit(train_data, train_labels)
m5.fit(train_data, train_labels)
m6.fit(train_data, train_labels)

# MODEL TESTING
print()
print('TEST SHAPE', np.shape(test_data))
print('MODEL TESTING...')
# res = logreg_clf.predict(test_data)
res2 = m2.predict(test_data)
# res3 = m3.predict(test_data)
res4 = m4.predict(test_data)
res5 = m5.predict(test_data)
res6 = m6.predict(test_data)

print()
# print('MODEL R2 SCORE', r2_score(test_labels, res))
print('MODEL2 R2 SCORE', accuracy_score(test_labels, res2))
# print('MODEL3 R2 SCORE', r2_score(test_labels, res3))
print('MODEL4 R2 SCORE', accuracy_score(test_labels, res4))
print('MODEL5 R2 SCORE', accuracy_score(test_labels, res5))
print('MODEL6 R2 SCORE', accuracy_score(test_labels, res6))

def getInputColor(label):
    if label == 1:
        return "blue"
    return "red"


def plotData():
    x = np.arange(0, 100, 1)
    y = np.arange(0, 100, 1)
    plt.xticks(x)
    plt.yticks(y)
    for i in range(200):
        for j in range(100):
            plt.scatter(j, inputs[i][j], c=getInputColor(outputs[i]))
    plt.show()

plotData()
