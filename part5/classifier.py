import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import csv


def getLabelValue(label):
    if label == "M":
        return 1
    return 0


with open('breast_cancer.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)


# LOAD DATASET
inputs = [[float(v) for v in d[2:]] for d in data]
outputs = [getLabelValue(d[1]) for d in data]
print(inputs[0])
print(outputs[0])
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
# MODEL TRAINING
print('MODEL TRAINING...')
logreg_clf.fit(train_data, train_labels)

# MODEL TESTING
print()
print('TEST SHAPE', np.shape(test_data))
print('MODEL TESTING...')
res = logreg_clf.predict(test_data)

print()
print('MODEL MEAN ACCURACY', accuracy_score(test_labels, res) * 100, '%')
