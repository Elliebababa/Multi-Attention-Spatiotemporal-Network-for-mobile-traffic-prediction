import numpy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import keras
import pickle
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential, Model
from keras.layers import Permute,Input,LSTM, Reshape,ConvLSTM2D, Activation, Dense, Conv3D, Add, TimeDistributed, Conv2D, Average
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import backend as K
K.set_image_data_format('channels_first')
'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config = config)
set_session(sess)
'''
def mape(y_true, y_pred):
    #mean absolute percentage error
    assert type(y_true) == type(y_pred)
    #assert np.shape(y_true) == np.shape(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))

def expand_dim_backend(x):
    x1 = K.expand_dims(x, -1)
    return x1
predstep = 1
nb_epoch = 0
bs = 2**10
modelbase = 'stn' # convlstm, stn
verbose=2
print("start!")

look_back = 6
import h5py
f = h5py.File('data_stn_Nov.h5','r')
tensor = f['X'].value[:]
tensorY = f['Y'].value[:]
slots = tensor.shape[0]
grids = tensor.shape[1]
print('X shape:',tensor.shape) #slots, grids, lookback(6),rows, cols (4320-6, 900, 6, 11, 11)
print('Y shape:',tensorY.shape) #slots, grids, 1

def multistep_prediction(test_input, model, pre_step = 1):
    #test_input (samples，T, n)
    pred = model.predict(test_input)
    target_seq = np.zeros((test_input[0].shape[0],1,1))
    target_seq[:,0,0] =  test_input[0][:,-1,0]

    decoded_seq = np.zeros((test_input[0].shape[0], pre_step, 1))

    for i in range(pre_step):
        if not decoder_aux is None:
            tmp = np.expand_dims(decoder_aux[:,i,], axis = 1)
            target_seq = [target_seq,tmp]
        else:
            target_seq = [target_seq]
        output, h, c = decoder_model.predict(target_seq + states_values)
        decoded_seq[:,i,0] = output[:,0,0]
        #update the target sequence of length 1
        target_seq = np.zeros((test_input[0].shape[0],1,1))
        target_seq[:,0,0] = output[:,0,0]
        #update states
        states_values = [h,c]
    return decoded_seq

def stn():
    look_back = 6
    wid = 11
    input = Input(shape=(look_back, wid, wid))
    input1 = Reshape(target_shape = (look_back, 1, wid, wid))(input)
    conlstm1 = ConvLSTM2D(filters = 3, kernel_size = (3,3), padding="same",return_sequences = True, activation = 'relu', recurrent_activation = 'relu')(input1)
    #flatten = Reshape(target_shape = (-1,))(conlstm1)
    #fc = Dense(1)(flatten)
    #output1 = Activation('sigmoid')(fc)
    
    input2 = Reshape(target_shape = (1, look_back, wid, wid))(input)
    con3d1 = Conv3D(filters = 3, kernel_size = (3,3,3), padding = "same", activation = 'relu')(input2)
    con3d1 = Conv3D(filters = 3, kernel_size = (3,3,3), padding = "same", activation = 'relu')(con3d1)
    con3d1 = Conv3D(filters = 3, kernel_size = (3,3,3), padding = "same", activation = 'relu')(con3d1)
    con3d1 = Permute((2,1,3,4))(con3d1)
    #flatten2 = Reshape(target_shape = (-1,))(con3d1)
    #fc2 = Dense(1)(flatten2)
    #output2 = Activation('sigmoid')(fc2)
    
    fusion1 = Average()([conlstm1,con3d1])
    conlstm2 = ConvLSTM2D(filters = 6, kernel_size = (3,3), padding="same",return_sequences = True, activation = 'relu', recurrent_activation = 'relu')(fusion1)

    fusion11 = Permute((2,1,3,4))(fusion1)
    con3d2 = Conv3D(filters = 6, kernel_size = (3,3,3), padding = "same", activation = 'relu')(fusion11)
    con3d2 = Conv3D(filters = 6, kernel_size = (3,3,3), padding = "same", activation = 'relu')(con3d2)
    con3d2 = Conv3D(filters = 6, kernel_size = (3,3,3), padding = "same", activation = 'relu')(con3d2)
    con3d2 = Permute((2,1,3,4))(con3d2)

    fusion2 = Average()([conlstm2,con3d2])

    flatten = Reshape(target_shape = (-1,))(fusion2) 
    fc = Dense(6)(flatten)
    fc = Dense(4)(fc)
    fc = Dense(1)(fc)
    output = Activation('sigmoid')(fc)
    
    model = Model(input = input, output = output, name = 'conlstm_conv3d')
    return model


def convlstm():
    look_back = 6
    wid = 11
    input = Input(shape=(look_back, wid, wid))
    input1 = Reshape(target_shape = (look_back, 1, wid, wid))(input)
    conlstm1 = ConvLSTM2D(filters = 3, kernel_size = (3,3), padding="same",return_sequences = True)(input1)
    flatten = Reshape(target_shape = (-1,))(conlstm1)
    
    fc = Dense(6)(flatten)
    fc = Dense(4)(fc)
    fc = Dense(1)(fc)
    output = Activation('sigmoid')(fc)
    
    model = Model(input = input, output = output, name = 'convlstm')
    return model

def cnnlstm():
    look_back = 6
    rows = 11
    input =  Input(shape=(look_back, rows, rows))
    input1 = Reshape(target_shape = (look_back, 1 , rows, rows))(input)
    wrapped = TimeDistributed(Conv2D(filters = 3,kernel_size = (5,5), strides=(1, 1), padding="same"))(input1)
    input1 = Reshape(target_shape = (look_back,-1))(wrapped)
    lstm = LSTM(64)(input1)
    output = Dense(1, activation='sigmoid')(lstm)
    model = Model(input = input, output = output, name ='cnnlstm')
    
    return model


#scaler
mmnX = MinMaxScaler()
mmn = MinMaxScaler()
tensor = mmnX.fit_transform(tensor.reshape((-1,6*11*11)))
tensorY = mmn.fit_transform(tensorY.reshape((-1,1)))


#testlen = 144*7#7 days length for testing
testlen = 1008 *900 #int(tensor.shape[0]/3)
# reshape input to be [samples, time steps, features]
trainX = tensor[:-testlen,:,]
testX = tensor[-testlen:,:,]
trainY = tensorY[:-testlen,:,]
testY = tensorY[-testlen:,:,]
trainX = np.reshape(trainX,(-1,6,11,11))
testX = np.reshape(testX,(-1,6,11,11))
trainY = np.reshape(trainY,(-1,1))
testY = np.reshape(testY,(-1,1))

print('trainX:',trainX.shape)
print('trainY:',trainY.shape)
print('testX:',testX.shape)
print('testY:',testY.shape)
print('finish loading')

if modelbase == 'convlstm':
    model = convlstm()
elif modelbase == 'stn':
    model = stn()
adam =Adam(lr=0.0002)
model.compile(optimizer= adam , loss='mean_squared_error')
model.summary()
fname_param = os.path.join('{}.best.h5'.format(modelbase))
if nb_epoch > 0:
    print((fname_param))
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_mean_squared_error', verbose=0, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    model.fit([trainX], trainY, epochs=nb_epoch, batch_size=bs, validation_split = 0.2, verbose=verbose, callbacks=[early_stopping, model_checkpoint])
    model.save_weights(fname_param)

#the evaluation will be divided into two parts
#the first is to evaluate over all the grids
#the second is to evaluate each grids seperately

#the first part is as before (remain the same)

model.load_weights(fname_param)

    
if predstep == 1:
    trainPredict = model.predict([trainX], verbose = verbose)
    testPredict = model.predict([testX], verbose = verbose)
else:
    trainPredict = multistep_prediction(trainX, model, predstep)
    testPredict = multistep_prediction(testX, model, predstep)


trainPredict = mmn.inverse_transform(trainPredict)
trainY = mmn.inverse_transform(trainY)
testPredict = mmn.inverse_transform(testPredict)
testY = mmn.inverse_transform(testY)

print(trainY.mean(), trainPredict.mean())
print(testY.mean(), testPredict.mean())
trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:, 0:1]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:], testPredict[:, 0:1]))
print('Test Score: %.2f RMSE' % (testScore), '\n')


trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:, 0:1]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:], testPredict[:, 0:1]))
print('Test Score: %.2f RMSE' % (testScore), '\n')

print('train mean:', numpy.mean(trainY), 'test mean:', numpy.mean(testY))
print('Train NRMSE: ', trainScore / numpy.mean(trainY))
print('Test NRMSE: ', testScore / numpy.mean(testY), '\n')

#the second part is to evaluate each grids

trainY = np.reshape(trainY, (-1, grids, 1))
testY = np.reshape(testY, (-1 , grids, 1))

trainPredict = np.reshape(trainPredict,( -1 , grids, 1))
testPredict = np.reshape(testPredict, (-1, grids, 1))

trainPredict = np.swapaxes(trainPredict,0,1)
testPredict = np.swapaxes(testPredict,0,1)

trainY = np.swapaxes(trainY, 0, 1)
testY = np.swapaxes(testY, 0, 1)

print('trainY shape:',trainY.shape)
trainScores = [(math.sqrt(mean_squared_error(trainY[grid,:], trainPredict[grid,:, 0:1])))/np.mean(trainY[grid,:]) for grid,_ in enumerate(trainPredict)]
testScores = [(math.sqrt(mean_squared_error(testY[grid,:], testPredict[grid,:, 0:1])))/np.mean(testY[grid,:]) for grid,_ in enumerate(testPredict)]

trainScores_rmse = [(math.sqrt(mean_squared_error(trainY[grid,:], trainPredict[grid,:, 0:1]))) for grid,_ in enumerate(trainPredict)]
testScores_rmse = [(math.sqrt(mean_squared_error(testY[grid,:], testPredict[grid,:, 0:1]))) for grid,_ in enumerate(testPredict)]


print(modelbase,' : ')
print('grid train nrmse mean:',np.mean(trainScores))
print('grid train nrmse std:',np.std(trainScores))
print('grid test nrmse mean:',np.mean(testScores))
print('grid test nrmse std:',np.std(testScores))
print('#'*10)
print('grid train rmse mean:',np.mean(trainScores_rmse))
print('grid train rmse std:',np.std(trainScores_rmse))
print('grid test rmse mean:',np.mean(testScores_rmse))
print('grid test rmse std:',np.std(testScores_rmse))





