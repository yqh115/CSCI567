import numpy as np
import json

input_file = 'mnist_subset.json'
output_file = 'knn_output.txt'

with open(input_file) as json_data:
    data = json.load(json_data)
train_set, valid_set, test_set = data['train'], data['valid'], data['test']
Xtrain = train_set[0]
ytrain = train_set[1]
Xval = valid_set[0]
yval = valid_set[1]
Xtest = test_set[0]
ytest = test_set[1]

Xtrain = np.array(Xtrain)
Xval = np.array(Xval)
Xtest = np.array(Xtest)

ytrain = np.array(ytrain)
yval = np.array(yval)
ytest = np.array(ytest)
# print(ytest)

print(Xtrain[1].shape)
print(Xtrain.shape)
print(np.linalg.norm(Xtrain[1] - Xtrain[3]))


# dists = np.ones((5, len(Xtrain)))
# for i in range(0, 5):
#     for j in range(0, len(Xtrain)):
#         dists[i][j] = np.linalg.norm(Xtrain[i] - Xtrain[j])
# print(np.argsort(dists, axis=1))
# print(dists)
a = np.array([0,1,1,2,5,5,0])
counts = np.bincount(a)
print(counts)
print(counts[0])
print(np.argmax(counts))