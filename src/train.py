from __future__ import print_function
#numerical package
import pandas as pd
from scipy import sparse
#system package
import click
import time
import h5py
import os
import math
import pickle
from sklearn.preprocessing import MinMaxScaler
#keras
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
#utils and metrics for training
from utils import *
import metrics
#model
from models import *
#allow growth
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#define parameters for training 
days_test = 7
T = 144
len_test = T * days_test
lr = 0.0002
look_back = 6 # look back of interval, the same as len_closeness
nb_epoch = 1
nb_epoch_cont = 0
patience = 5  # early stopping patience
batch_size = 2**10
verbose = 1
#model for training
modelbase = 'seq2seq'
m = 64 #hidden layer of MAModel

print('\n','='*5, ' configuration ', '='*5)
print('\nlr: %.5f, lookback: %d, nb_epoch: %d, patience:%d, nb_epoch_cont: %d, batch size:%d'%(lr, look_back, nb_epoch, patience, nb_epoch_cont, batch_size))
print('\n','='*15,'\n')

def scale_data(data, ran=(0, 1)):
    shape = data.shape
    mmn = MinMaxScaler(feature_range = ran)
    data_scale = mmn.fit_transform(data.reshape(shape[0],-1))
    data_scale = data_scale.reshape(shape)
    return data_scale, mmn

def display_evaluation(y_pred, y_true, info = 'evaluation:'):
    assert(np.shape(y_pred) == np.shape(y_true))
    score_mse = np.mean(np.power(y_pred-y_true,2))
    score_rmse = np.power(score_mse, 0.5)
    score_nrmse = score_rmse / np.mean(y_true)
    score_mape = np.mean(np.abs((y_true - y_pred)/y_true))
    print('{} mse:{}, rmse:{}, nrmse:{}, mape:{}'.format(info,score_mse, score_rmse, score_nrmse, score_mape))
    
    
def build_model(modelbase = modelbase):
    if modelbase == 'seq2seq':
        model, encoder_model, decoder_model = seq2seq()
    if modelbase == 'MAModel':
        a = MAModel()
        model = a.build_model(input_dim = 5)
        encoder_model = None
        decoder_model = None
    adam = Adam(lr = lr)
    model.compile(loss = 'mse', optimizer = adam, metrics = [metrics.rmse, metrics.mape, metrics.ma])
    model.summary()
    
    return model, encoder_model, decoder_model

import click
@click.command()
@click.option('--modelbase',default= modelbase, help='seq2seq, MAModel')
def main(modelbase):
    modelbase = modelbase
    # load data and make dataset
    print('loading data...')
    ts = time.time()
    train_encoder_input, train_encoder_input_aux, train_decoder_input, train_decoder_input_his, train_decoder_target, test_encoder_input, test_encoder_input_aux, test_decoder_input_his, test_decoder_target = loadData()

    print('Train set shape: \ntrain_encoder_input:{}, \ntrain_encoder_input_aux:{}, \ntrain_decoder_input:{}, \ntrain_decoder_input_his:{}, \ntrain_decoder_target:{}'.format(
       train_encoder_input.shape, train_encoder_input_aux.shape, train_decoder_input.shape, train_decoder_input_his.shape, train_decoder_target.shape ))
    print('Test set shape: \ntest_encoder_input:{}, \ntest_encoder_input_aux:{}, \ntest_decoder_input_his:{}, \ntest_decoder_target:{}'.format(
        test_encoder_input.shape, test_encoder_input_aux.shape, test_decoder_input_his.shape, test_decoder_target.shape ))

  
    #test decoder input
    test_decoder_input = np.zeros(test_decoder_target.shape)
    test_decoder_input[:,:,1:,] = test_decoder_target[:,:,:-1]
    test_decoder_input[:,:,0,] = test_encoder_input[:,:,-1,0:1]

    print('\n\nstacking data and scaling data to (0,1)...')
    train_encoder_input = np.vstack(train_encoder_input)
    test_encoder_input = np.vstack(test_encoder_input)
    encoder_input = np.concatenate([train_encoder_input, test_encoder_input],axis = 0)
    encoder_input, encoder_input_mmn = scale_data(encoder_input)
    train_encoder_input = encoder_input[:train_encoder_input.shape[0]]
    test_encoder_input = encoder_input[train_encoder_input.shape[0]:]
    
    train_encoder_input_aux = np.vstack(train_encoder_input_aux)
    test_encoder_input_aux = np.vstack(test_encoder_input_aux)
    encoder_input_aux = np.concatenate([train_encoder_input_aux, test_encoder_input_aux],axis = 0)
    encoder_input_aux, encoder_input_aux_mmn = scale_data(encoder_input_aux)
    train_encoder_input_aux = encoder_input_aux[:train_encoder_input_aux.shape[0]]
    test_encoder_input_aux = encoder_input_aux[train_encoder_input_aux.shape[0]:]
    
    #concate x and aux
    #encoder_input
    if modelbase == 'MAModel':
        train_encoder_input = np.concatenate([train_encoder_input, train_encoder_input_aux],axis = -1)
        test_encoder_input = np.concatenate([test_encoder_input, test_encoder_input_aux],axis = -1)
    
    #decoder input
    train_decoder_input = np.vstack(train_decoder_input)
    test_decoder_input = np.vstack(test_decoder_input)
    decoder_input = np.concatenate([train_decoder_input, test_decoder_input],axis = 0)
    decoder_input, decoder_input_mmn = scale_data(decoder_input)
    train_decoder_input = decoder_input[:train_decoder_input.shape[0]]
    test_decoder_input = decoder_input[train_decoder_input.shape[0]:]
    
    #decoder_input_his
    train_decoder_input_his = np.vstack(train_decoder_input_his)
    test_decoder_input_his = np.vstack(test_decoder_input_his)
    
    #decoder_target
    train_decoder_target = np.vstack(train_decoder_target)
    test_decoder_target = np.vstack(test_decoder_target)
    decoder_target = np.concatenate([train_decoder_target, test_decoder_target],axis = 0)
    decoder_target, decoder_target_mmn = scale_data(decoder_target)
    train_decoder_target = decoder_target[:train_decoder_target.shape[0]]
    test_decoder_target = decoder_target[train_decoder_target.shape[0]:]
    
    print('Train set shape: \ntrain_encoder_input:{}, \ntrain_encoder_input_aux:{}, \ntrain_decoder_input:{}, \ntrain_decoder_input_his:{}, \ntrain_decoder_target:{}'.format(
        train_encoder_input.shape, train_encoder_input_aux.shape, train_decoder_input.shape, train_decoder_input_his.shape, train_decoder_target.shape ))
    print('Test set shape: \ntest_encoder_input:{}, \ntest_encoder_input_aux:{}, \ntest_decoder_input_his:{}, \ntest_decoder_target:{}'.format(
        test_encoder_input.shape, test_encoder_input_aux.shape, test_decoder_input_his.shape, test_decoder_target.shape ))
    print("\n elapsed time (loading data and making dataset): %.3f seconds\n" % (time.time() - ts))

    #==== compiling model ==================================================================================    
    
    print('=' * 10)
    print("compiling model...")
    ts = time.time()
    #build model
    model, encoder_model, decoder_model = build_model(modelbase)
    #filename for saving the best model
    fname_param = os.path.join('{}.best.h5'.format(modelbase))
    #callbacks
    early_stopping = EarlyStopping(monitor='val_root_mean_square_error', patience=patience, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_root_mean_square_error', verbose=0, save_best_only=True, mode='min')
    tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 0, write_graph = True, write_images = False, embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)
    print("\ncompile model elapsed time (compiling model): %.3f seconds\n" %(time.time() - ts))
    
    #==== training model ===================================================================================
    
    print('=' * 10)
    print("training model...")
    ts = time.time()
    #history = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, model_checkpoint,tensor_board], verbose=verbose)
    callbacks = [early_stopping, model_checkpoint,tensor_board]
    if modelbase =='seq2seq':
        history = model.fit([train_encoder_input, train_decoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.2, callbacks = callbacks)
    if modelbase =='MAModel':
        s0_train = h0_train = np.zeros((train_encoder_input.shape[0],m))
        enc_att = np.ones((train_encoder_input.shape[0],1,train_encoder_input.shape[2]))
        history = model.fit([train_encoder_input,h0_train,s0_train,enc_att,train_decoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.2, callbacks = callbacks)
    model.save_weights(fname_param)
    print("\ntrain model elapsed time (training): %.3f seconds\n" % (time.time() - ts))
    
    #==== evaluating model ===================================================================================
    mmn = decoder_target_mmn
    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(fname_param)
    #train score
    if modelbase =='seq2seq':
        train_pred =  model.predict([train_encoder_input, train_decoder_input], verbose = verbose, batch_size = batch_size)
    if modelbase =='MAModel':
        s0_train = h0_train = np.zeros((train_encoder_input.shape[0],m))
        enc_att = np.ones((train_encoder_input.shape[0],1,train_encoder_input.shape[2]))
        train_pred =  model.predict([train_encoder_input,h0_train,s0_train,enc_att, train_decoder_input], verbose = verbose, batch_size = batch_size)
    
    print('train_pred shape',train_pred.shape)
    train_pred_orig = mmn.inverse_transform(train_pred.reshape(train_pred.shape[0],-1))
    train_decoder_target_orig = mmn.inverse_transform(train_decoder_target.reshape(train_decoder_target.shape[0],-1))
    display_evaluation(train_pred_orig,train_decoder_target_orig,'Train score')

    #test score
    if modelbase =='seq2seq':
        test_pred = decoder_prediction([test_encoder_input], encoder_model, decoder_model)
    if modelbase =='MAModel':
        s0_test = h0_test = np.zeros((test_encoder_input.shape[0],m))
        enc_att = np.ones((test_encoder_input.shape[0],1,test_encoder_input.shape[2]))
        
        test_pred = model.predict([test_encoder_input,h0_train,s0_train,enc_att, test_decoder_input], verbose = verbose, batch_size = batch_size)
    print('test_pred shape: ',test_pred.shape)
    test_pred_orig = mmn.inverse_transform(test_pred.reshape(test_pred.shape[0],-1))
    test_decoder_target_orig = mmn.inverse_transform(test_decoder_target.reshape(test_decoder_target.shape[0],-1))
    display_evaluation(test_pred_orig,test_decoder_target_orig,'Test score')
    print("\nevaluate model elapsed time (eval): %.3f seconds\n" % (time.time() - ts))
    
if __name__ == '__main__':
    main()




    
         
