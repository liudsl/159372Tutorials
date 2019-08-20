import pylab as pl
import numpy as np
import pcn
import pickle, gzip

# Read the dataset in (code from sheet)
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
f.close()
train_in = train_set[0][:200]
train_out = train_set[1][:200]
test_in = test_set[0][:200]
test_out = test_set[1][:200]


pl.imshow(np.reshape(train_set[0][0, :], [28, 28]))
print("The correct digit for the first image in the train set is :", train_set[1][0])
pl.show()


p = pcn.pcn(train_in, train_out)
p.pcntrain(train_in, train_out, 0.25, 100)
p.confmat(test_in, test_out)

