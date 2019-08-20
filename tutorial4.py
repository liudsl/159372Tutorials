import gzip
import pickle
import numpy as np
import pcn

# Read the dataset in (code from sheet)
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
f.close()


train_in = train_set[0][:200, :]
train_out = np.zeros((200, 10))

for x in range(200):
    train_out[x, train_set[1][x]] = 1

test_in = test_set[0][:200, :]
test_out = np.zeros((200, 10))

for x in range(200):
    test_out[x, test_set[1][x]] = 1

# pl.imshow(np.reshape(train_set[0][0, :], [28, 28]))
# print("The correct digit for the first image in the train set is :", train_set[1][0])
# pl.show()

p = pcn.pcn(train_in, train_out)
p.pcntrain(train_in, train_out, 0.25, 100)
p.confmat(train_in, train_out)
p.confmat(test_in, test_out)
