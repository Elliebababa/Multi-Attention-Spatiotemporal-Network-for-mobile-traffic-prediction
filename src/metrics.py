import numpy as np
from keras import backend as K

def mse(y_true, y_pred):
    #print('y_true',y_true)
    #print('y_pred',y_pred)
    assert type(y_true) == type(y_pred)
    #assert (y_true) == np.shape(y_pred)
    return K.mean(K.square(y_pred - y_true))


def rmse(y_true, y_pred):
    #print('y_true',y_true)
    #print('y_pred',y_pred)
    assert type(y_true) == type(y_pred)
    #assert y_true.shape == y_pred.shape
    return mse(y_true, y_pred) ** 0.5


# aliases
#mse = MSE = mean_squared_error
#rmse = RMSE = root_mean_square_error

def mape(y_true, y_pred):
    #mean absolute percentage error
    assert type(y_true) == type(y_pred)
    #assert np.shape(y_true) == np.shape(y_pred)
    return K.mean(K.abs((y_true - y_pred)/y_true))

def ma(y_true, y_pred):
    #1 - mape
    return 1 - mape(y_true, y_pred)

