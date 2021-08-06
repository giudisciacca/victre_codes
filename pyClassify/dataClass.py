
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import h5py

def load_file(path):
    fData = h5py.File(path, 'r')
    features = numpy.array(fData.get('features'))
    labels = numpy.array(fData.get('labels'))
    return features,labels


class Dataset(object):
    def __init__(self,pathname):
        self._features, self._labels = load_file(pathname);
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = numpy.shape(self.features)[0];
        return

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]


class fullDataset(object):
    def __init__(self,pathname):
        self.pathname = pathname;
        self.train = Dataset(pathname+'.mat');
        self.test = Dataset(pathname+'_t.mat');
        self.valid = Dataset(pathname+'_v_t.mat');


