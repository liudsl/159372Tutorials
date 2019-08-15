import pylab as pl
import numpy as np
import pcn
import pickle, gzip

# Read the dataset in (code from sheet)
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
f.close()

pl.imshow(np.reshape(train_set[0][0,:], [28, 28]))
print("The correct digit for the first image in the train set is :", train_set[1][0])
pl.show()



