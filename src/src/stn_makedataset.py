import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import pickle
import math
import keras
import time
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import h5py as h5

f1 =  h5.File('../../../data/processed/Nov_traffic_data.h5','r') 
data = f1['data'].value

radius = 5

predstep = 3

def createdataset(i,j, look_back=1,predstep = predstep):
    dataX, dataY = [], []
    for t in range(len(data) - look_back - predstep + 1):
        a = data[t:(t + look_back),0,i-radius:i+radius+1,j-radius:j+radius+1]
        dataX.append(a)
        dataY.append(data[t+look_back:t+look_back+predstep,0:1,i,j])
    return dataX, dataY

look_back = 6
n = data.shape[0] - look_back - predstep + 1#time slots
m = 900 # grid Num
wid = radius*2 + 1
tensor = [None]*n
for i in range(len(tensor)):
    tensor[i] = [0.0]*m
for i in range(len(tensor)):
    for j in range(len(tensor[0])):
        tensor[i][j] = np.zeros((look_back,wid,wid),dtype='float32')
tensor = np.array(tensor)
print('X shape',tensor.shape)

tensorY = [None]*n
for i in range(len(tensorY)):
    tensorY[i] = [0.0]*m
for i in range(len(tensorY)):
    for j in range(len(tensorY[0])):
        tensorY[i][j] = np.zeros((predstep,1),dtype='float32')
tensorY = np.array(tensorY)
print('Y shape',tensorY.shape) 

ind = 0
for i in range(40,70):
    for j in range(40,70):
        print('grid(%d,%d)...'%(i,j))
        X,Y = createdataset(i,j,look_back)
        tensor[:,ind,] = np.array(X)        
        tensorY[:,ind,] = np.array(Y)
        ind += 1
        print('finish loading this grid...')


f5 = h5.File('data_stn_Nov_pred{}.h5'.format(predstep))
f5.create_dataset('X',data = tensor)
f5.create_dataset('Y',data = tensorY)
f5.close()






