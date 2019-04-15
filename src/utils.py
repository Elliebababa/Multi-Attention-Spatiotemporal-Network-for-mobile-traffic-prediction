import os
import h5py
import numpy as np
import pickle
class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        #X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


def loadData(fname):
    f = h5py.File(fname, 'r')
    print('loading from file {}, with keys:{}'.format(fname, list(f.keys())))
    #['test_decoder_input_his', 'test_decoder_target', 'test_encoder_input', 'test_encoder_input_aux', 'train_decoder_input', 'train_decoder_input_his', 'train_decoder_target', 'train_encoder_input', 'train_encoder_input_aux']
    if 'test_decoder_input_his' in f.keys():
      test_decoder_input_his = f['test_decoder_input_his'].value
    else:
      test_decoder_input_his = np.zeros((1,1))
    if 'test_neighbor_values' in f.keys() and 'test_neighbor_weights' in f.keys():
      test_neighbor_values = f['test_neighbor_values'].value
      test_neighbor_weights = f['test_neighbor_weights'].value
    else:
      test_neighbor_values = np.zeros((1,1))
      test_neighbor_weights = np.zeros((1,1))
    test_decoder_target = f['test_decoder_target'].value
    test_encoder_input = f['test_encoder_input'].value
    test_encoder_input_aux = f['test_encoder_input_aux'].value
    
    train_decoder_input = f['train_decoder_input'].value
    if 'train_decoder_input_his' in f.keys():
      train_decoder_input_his = f['train_decoder_input_his'].value
    else:
      train_decoder_input_his = np.zeros((1,1))
    if 'train_neighbor_values' in f.keys() and 'train_neighbor_weights' in f.keys():
      train_neighbor_values = f['train_neighbor_values'].value
      train_neighbor_weights = f['train_neighbor_weights'].value
    else:
      train_neighbor_values = np.zeros((1,1))
      train_neighbor_weights = np.zeros((1,1))
      
    if 'train_semantic_input_data' in f.keys() and 'test_semantic_input_data' in f.keys():
      train_semantic_input_data = f['train_semantic_input_data'].value
      test_semantic_input_data = f['test_semantic_input_data'].value
    else:
      train_semantic_input_data = np.zeros((1,1))
      test_semantic_input_data = np.zeros((1,1))
    train_decoder_target = f['train_decoder_target'].value
    train_encoder_input = f['train_encoder_input'].value
    train_encoder_input_aux = f['train_encoder_input_aux'].value
    
    
    f.close()
    return train_encoder_input, train_encoder_input_aux, train_decoder_input, train_decoder_input_his, train_decoder_target, test_encoder_input, test_encoder_input_aux, test_decoder_input_his, test_decoder_target, \
        train_neighbor_values, test_neighbor_values, train_neighbor_weights, test_neighbor_weights, train_semantic_input_data, test_semantic_input_data
    
