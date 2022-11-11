import numpy as np
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

print(inputs[0])
print(inputs[1])
print(inputs[2])
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
m3 = DecisionTreeClassifier()
# MODEL TRAINING
print('MODEL TRAINING...')
logreg_clf.fit(train_data, train_labels)
m2.fit(train_data, train_labels)
m3.fit(train_data, train_labels)

# MODEL TESTING
print()
print('TEST SHAPE', np.shape(test_data))
print('MODEL TESTING...')
res = logreg_clf.predict(test_data)
res2 = m2.predict(test_data)
res3 = m3.predict(test_data)

print()
print('MODEL MEAN ACCURACY', accuracy_score(test_labels, res) * 100, '%')
print('MODEL2 MEAN ACCURACY', accuracy_score(test_labels, res2) * 100, '%')
print('MODEL3 MEAN ACCURACY', accuracy_score(test_labels, res3) * 100, '%')
