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
from sklearn.metrics import mean_squared_error
#keras
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
#utils and metrics for training
from utils import *
import metrics
#model
from models import *
#code to fix keras of tf bug
#allow growth
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#define parameters for training 
lr = 0.0002
look_back = 6 # look back of interval, the same as len_closeness
nb_epoch = 100
patience = 100  # early stopping patience, for I find early stopping did not contribute to model training in this experiment, I will ignore this parameter
batch_size = 2**10
verbose = 2
#model for training
modelbase = 'MASTNN-spatioatt-auxatt' #RNN , lstm, lstm_aux, seq2seq, seq2seq_aux , MASTNN, MASTNN-spatioatt, MASTNN-auxatt, MASTNN-spatioatt-auxatt, MASTNN-decodelast, MASTNN-decodemean
m = 64 #hidden layer of MAModel
predstep = 1
# for testing the model training performance
cuttrain = False
cuttrain_len = 1008*3 # for testing the model training performance

def scale_data(data, ran=(0, 1)):
    #reahpe the data into (samples, dim) and then reshape it back
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
    elif modelbase == 'seq2seq_aux':
        model, encoder_model, decoder_model = seq2seq(input_dim = 5)
    elif modelbase == 'lstm':
        model = lstm()
        encoder_model = None
        decoder_model = model
    elif modelbase == 'lstm_aux':
        model = lstm(input_dim = 5)
        encoder_model = None
        decoder_model = model
    else:
        encoder_model = None
        decoder_model = None
        if modelbase == 'MASTNN':
            a = MASTNN(predT = predstep,T = look_back)
            model = a.build_model(input_dim = 5)
            #MASTNN-spatioatt, MASTNN-auxatt, MASTNN-decodelast, MASTNN-decodemean
        elif modelbase == 'MASTNN-spatioatt':
            a = MASTNN(predT = predstep,global_att = False,T = look_back)
            model = a.build_model(input_dim = 5)
        elif modelbase == 'MASTNN-auxatt':
            a = MASTNN(predT = predstep,T = look_back,aux_att = False)
            model = a.build_model(input_dim = 5)
        elif modelbase == 'MASTNN-spatioatt-auxatt':
            a = MASTNN(predT = predstep,T = look_back,global_att = False,aux_att = False)
            model = a.build_model(input_dim = 5)
        elif modelbase == 'MASTNN-decodelast':
            a = MASTNN(predT = predstep,T = look_back)
            model = a.build_model(input_dim = 5)
        elif modelbase == 'MASTNN-decodemean':
            a = MASTNN(predT = predstep, T = look_back)
            model = a.build_model(input_dim = 5)
    adam = Adam(lr = lr)
    model.compile(loss = 'mse', optimizer = adam, metrics = [metrics.rmse, metrics.mape, metrics.ma])
    model.summary()
    return model, encoder_model, decoder_model

import click
@click.command()
@click.option('--modelbase',default= modelbase, help='RNN , lstm, lstm_aux, seq2seq, seq2seq_aux , MASTNN, MASTNN-spatioatt, MASTNN-auxatt, MASTNN-decodelast, MASTNN-decodemean') #with - means having the module in the model
def main(modelbase):
    modelbase = modelbase
    
    #print configuration
    print('\n','='*5, ' configuration ', '='*5)
    print('model: ', modelbase)
    print('\nlr: %.5f, lookback: %d, nb_epoch: %d, patience:%d, predstep: %d, batch size:%d'%(lr, look_back, nb_epoch, patience, predstep, batch_size))
    print('\n','='*15,'\n')

    # load data and make dataset
    print('loading data...')
    ts = time.time()
    fname = 'train_test_set_en6_de{}_Nov_neighbor.h5'.format(predstep) 
    train_encoder_input, train_encoder_input_aux, train_decoder_input, train_decoder_input_his, train_decoder_target, test_encoder_input, test_encoder_input_aux, test_decoder_input_his, test_decoder_target, \
        train_neighbor_values, test_neighbor_values, train_neighbor_weights, test_neighbor_weights , train_semantic_input, test_semantic_input = loadData(fname)

    #cut train len for testing training performance
    if cuttrain:
        train_encoder_input, train_encoder_input_aux, train_decoder_input,  train_decoder_target, train_neighbor_values, train_neighbor_weights, train_semantic_input = train_encoder_input[:,:-cuttrain_len,], train_encoder_input_aux[:,:-cuttrain_len,], train_decoder_input[:,:-cuttrain_len,], train_decoder_target[:,:-cuttrain_len,], train_neighbor_values[:,:-cuttrain_len,], train_neighbor_weights[:,:-cuttrain_len,], train_semantic_input[:,:-cuttrain_len,]

    print('Train set shape: \ntrain_encoder_input:{}, \ntrain_encoder_input_aux:{}, \ntrain_decoder_input:{}, \ntrain_decoder_input_his:{}, \ntrain_decoder_target:{}'.format(
       train_encoder_input.shape, train_encoder_input_aux.shape, train_decoder_input.shape, train_decoder_input_his.shape, train_decoder_target.shape ))
    print('Test set shape: \ntest_encoder_input:{}, \ntest_encoder_input_aux:{}, \ntest_decoder_input_his:{}, \ntest_decoder_target:{}, \ntest_semantic_input:{}'.format(
        test_encoder_input.shape, test_encoder_input_aux.shape, test_decoder_input_his.shape, test_decoder_target.shape , test_semantic_input.shape))

    gridNum, trainSlots, lookback, _ = train_encoder_input.shape #gridNum, trainSlots, lookback, input_dim
    _, _, predSteps, _ = train_decoder_target.shape #gridNum, trainSlots, predStep, decode_input_dim
    _, testSlots, _, _ = test_encoder_input.shape #gridNum, testSlots, lookback, input_dim
    
    #test decoder input
    test_decoder_input = np.zeros(test_decoder_target.shape)
    test_decoder_input[:,:,0,] = test_encoder_input[:,:,-1,0:1]
    #test_decoder_input[:,:,1:,] = test_decoder_target[:,:,:-1]

    print('\n\nstacking data and scaling data to (0,1)...')
    train_encoder_input = np.vstack(np.swapaxes(train_encoder_input,0,1))
    test_encoder_input = np.vstack(np.swapaxes(test_encoder_input,0,1))
    encoder_input = np.concatenate([train_encoder_input, test_encoder_input],axis = 0)
    encoder_input, encoder_input_mmn = scale_data(encoder_input)
    train_encoder_input = encoder_input[:train_encoder_input.shape[0]]
    test_encoder_input = encoder_input[train_encoder_input.shape[0]:]
    
    train_encoder_input_aux = np.vstack(np.swapaxes(train_encoder_input_aux, 0 , 1))
    test_encoder_input_aux = np.vstack(np.swapaxes(test_encoder_input_aux, 0, 1))
    encoder_input_aux = np.concatenate([train_encoder_input_aux, test_encoder_input_aux],axis = 0)
    encoder_input_aux, encoder_input_aux_mmn = scale_data(encoder_input_aux)
    train_encoder_input_aux = encoder_input_aux[:train_encoder_input_aux.shape[0]]
    test_encoder_input_aux = encoder_input_aux[train_encoder_input_aux.shape[0]:]
    
    #concate x and aux
    #encoder_input
    if not modelbase == 'lstm' and not modelbase == 'seq2seq':
        train_encoder_input = np.concatenate([train_encoder_input, train_encoder_input_aux],axis = -1)
        test_encoder_input = np.concatenate([test_encoder_input, test_encoder_input_aux],axis = -1)
        
    #neighbor_weights and neighbor_values
    if modelbase[:6] == 'MASTNN':
        train_neighbor_values = np.vstack(np.swapaxes(train_neighbor_values,0,1))
        test_neighbor_values = np.vstack(np.swapaxes(test_neighbor_values,0,1))
        neighbor_values = np.concatenate([train_neighbor_values, test_neighbor_values],axis = 0)
        neighbor_values, neighbor_values_mmn = scale_data(neighbor_values)
        train_neighbor_values = neighbor_values[:train_neighbor_values.shape[0]]
        test_neighbor_values = neighbor_values[train_neighbor_values.shape[0]:]
        train_neighbor_weights = np.vstack(np.swapaxes(train_neighbor_weights,0,1))
        test_neighbor_weights = np.vstack(np.swapaxes(test_neighbor_weights,0,1))
        print('train_neighbor_values:{}, \ntest_neighbor_values:{}'.format(train_neighbor_values.shape, test_neighbor_values.shape))
        print('train_neighbor_weights:{}, \ntest_neighbor_weights:{}'.format(train_neighbor_weights.shape, test_neighbor_weights.shape))

    #decoder input
    train_decoder_input = np.vstack(np.swapaxes(train_decoder_input,0,1))
    test_decoder_input = np.vstack(np.swapaxes(test_decoder_input,0,1))
    decoder_input = np.concatenate([train_decoder_input, test_decoder_input],axis = 0)
    decoder_input, decoder_input_mmn = scale_data(decoder_input)
    train_decoder_input = decoder_input[:train_decoder_input.shape[0]]
    test_decoder_input = decoder_input[train_decoder_input.shape[0]:]
        
    #decoder_target
    train_decoder_target = np.vstack(np.swapaxes(train_decoder_target,0,1))
    test_decoder_target = np.vstack(np.swapaxes(test_decoder_target,0,1))
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
    fname_param = os.path.join('testmodel/{}_look{}_pred{}.best.h5'.format(modelbase, lookback, predstep))
    #callbacks
    early_stopping = EarlyStopping(monitor='val_rmse', patience=patience, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
    tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 0, write_graph = True, write_images = False, embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)
    print("\ncompile model elapsed time (compiling model): %.3f seconds\n" %(time.time() - ts))
    
    #==== training model ===================================================================================
    
    print('=' * 10)
    if nb_epoch > 0:
    #train model when nb_epoch is set larger than 0
    #'RNN , lstm, lstm_aux, seq2seq, seq2seq_aux , MASTNN, MASTNN-spatioatt, MASTNN-auxatt, MASTNN-decodelast, MASTNN-decodemean'
        print("training model...")
        ts = time.time()
        #history = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, model_checkpoint,tensor_board], verbose=verbose)
        callbacks = [early_stopping, model_checkpoint,tensor_board]
        if modelbase in ['seq2seq', 'seq2seq_aux']:
            history = model.fit([train_encoder_input, train_decoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.2, callbacks = callbacks)
        elif modelbase in ['lstm', 'lstm_aux']:
            history = model.fit([train_encoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.2, callbacks = callbacks)
        elif modelbase =='MASTNN-spatioatt':
            s0_train = h0_train = np.zeros((train_encoder_input.shape[0],m))
            enc_att = np.ones((train_encoder_input.shape[0],1,train_encoder_input.shape[2]))
            history = model.fit([train_encoder_input,h0_train,s0_train,enc_att,train_decoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.2, callbacks = callbacks)
        else:
            #model = Model([encoder_inputs_local, encoder_inputs_global_value,encoder_inputs_global_weight, h0, s0, enc_att_local, enc_att_global, decoder_inputs], output)
            s0_train = h0_train = np.zeros((train_encoder_input.shape[0],m))
            enc_att = np.ones((train_encoder_input.shape[0],1,train_encoder_input.shape[2]))
            enc_att_glo = np.ones((train_neighbor_values.shape[0],1,train_neighbor_values.shape[2]))
            #model.trainmode = True
            history = model.fit([train_encoder_input, train_neighbor_values, train_neighbor_weights, h0_train, s0_train, enc_att, enc_att_glo, train_decoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.2, callbacks = callbacks)
            #model.trainmode = False
        model.save_weights(fname_param)
        print("\ntrain model elapsed time (training): %.3f seconds\n" % (time.time() - ts))
    
    #==== evaluating model ===================================================================================
    
    mmn = decoder_target_mmn
    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(fname_param)
    #train score
    if modelbase in ['seq2seq', 'seq2seq_aux']:
        train_pred =  model.predict([train_encoder_input, train_decoder_input], verbose = verbose, batch_size = batch_size)
    elif modelbase in ['lstm', 'lstm_aux']:
        train_pred =  lstm_prediction([train_encoder_input], model, pre_step = predstep)
    elif modelbase =='MASTNN-spatioatt':
        s0_train = h0_train = np.zeros((train_encoder_input.shape[0],m))
        enc_att = np.ones((train_encoder_input.shape[0],1,train_encoder_input.shape[2]))
        train_pred =  model.predict([train_encoder_input,h0_train,s0_train,enc_att, train_decoder_input], verbose = verbose, batch_size = batch_size)
    else:
    #model = Model([encoder_inputs_local, encoder_inputs_global_value,encoder_inputs_global_weight, h0, s0, enc_att_local, enc_att_global, decoder_inputs], output)
        s0_train = h0_train = np.zeros((train_encoder_input.shape[0],m))
        enc_att = np.ones((train_encoder_input.shape[0],1,train_encoder_input.shape[2]))
        enc_att_glo = np.ones((train_neighbor_values.shape[0],1,train_neighbor_values.shape[2]))
        train_pred=model.predict([train_encoder_input, train_neighbor_values, train_neighbor_weights, h0_train, s0_train, enc_att, enc_att_glo, train_decoder_input],verbose = verbose, batch_size=batch_size)
    
    print('train_pred shape',train_pred.shape)
    train_pred_orig = mmn.inverse_transform(train_pred.reshape(train_pred.shape[0],-1))
    train_decoder_target_orig = mmn.inverse_transform(train_decoder_target.reshape(train_decoder_target.shape[0],-1))
    display_evaluation(train_pred_orig,train_decoder_target_orig,'Train score')

    #test score
    if modelbase in ['seq2seq', 'seq2seq_aux']:
        test_pred = decoder_prediction([test_encoder_input], encoder_model, decoder_model, pre_step = predstep)
        #test_pred = model.predict([test_encoder_input, test_decoder_input], encoder_model, decoder_model)
    elif modelbase in ['lstm', 'lstm_aux']:
        test_pred = lstm_prediction([test_encoder_input], model, pre_step = predstep)
    elif modelbase =='MASTNN-spatioatt':
        s0_test = h0_test = np.zeros((test_encoder_input.shape[0],m))
        enc_att = np.ones((test_encoder_input.shape[0],1,test_encoder_input.shape[2]))
        test_pred = model.predict([test_encoder_input,h0_train,s0_train,enc_att, test_decoder_input], verbose = verbose, batch_size = batch_size)
    elif modelbase =='MASTNN':
        s0_test = h0_test = np.zeros((test_encoder_input.shape[0],m))
        enc_att = np.ones((test_encoder_input.shape[0],1,test_encoder_input.shape[2]))
        enc_att_glo = np.ones((test_neighbor_values.shape[0],1,test_neighbor_values.shape[2]))
        test_pred = model.predict([test_encoder_input, test_neighbor_values, test_neighbor_weights, h0_test, s0_test, enc_att, enc_att_glo, test_decoder_input], verbose = verbose, batch_size = batch_size)
    print('test_pred shape: ',test_pred.shape)
    test_pred_orig = mmn.inverse_transform(test_pred.reshape(test_pred.shape[0],-1))
    test_decoder_target_orig = mmn.inverse_transform(test_decoder_target.reshape(test_decoder_target.shape[0],-1))
    display_evaluation(test_pred_orig,test_decoder_target_orig,'Test score')
    print("\nevaluate model elapsed time (eval): %.3f seconds\n" % (time.time() - ts))
    
#==========================================new evaluation========================================
    #the second part is to evaluate each grids

    train_pred_orig = np.reshape(train_pred_orig,(trainSlots,gridNum,predstep))
    train_decoder_target_orig = np.reshape(train_decoder_target_orig,(trainSlots,gridNum,predstep))
    
    test_pred_orig = np.reshape(test_pred_orig,(testSlots,gridNum,predstep))
    test_decoder_target_orig = np.reshape(test_decoder_target_orig,(testSlots,gridNum,predstep))
    
    trainPredict = np.swapaxes(train_pred_orig,0,1)
    testPredict = np.swapaxes(test_pred_orig,0,1)

    trainY = np.swapaxes(train_decoder_target_orig, 0, 1)
    testY = np.swapaxes(test_decoder_target_orig, 0, 1)

    np.save('testdata/{}_en{}_de_{}_pred.npy'.format(modelbase, lookback, predstep),testPredict)
    np.save('testdata/{}_en_{}_de_{}_test.npy'.format(modelbase, lookback, predstep),testY)

    print('trainY shape:',trainY.shape)
    print('trainPredict.shape',trainPredict.shape)
    trainScores = [(math.sqrt(mean_squared_error(trainY[grid,:].flatten(), trainPredict[grid,:].flatten())))/np.mean(trainY[grid,:].flatten()) for grid,_ in enumerate(trainPredict)]
    testScores = [(math.sqrt(mean_squared_error(testY[grid,:].flatten(), testPredict[grid,:].flatten())))/np.mean(testY[grid,:].flatten()) for grid,_ in enumerate(testPredict)]
    
    trainScores_rmse = [(math.sqrt(mean_squared_error(trainY[grid,:].flatten(), trainPredict[grid,:].flatten()))) for grid,_ in enumerate(trainPredict)]
    testScores_rmse = [(math.sqrt(mean_squared_error(testY[grid,:].flatten(), testPredict[grid,:].flatten()))) for grid,_ in enumerate(testPredict)]
    
    trainScores_mae = [(np.mean(np.abs(trainY[grid,:].flatten()- trainPredict[grid,:].flatten()))) for grid,_ in enumerate(trainPredict)]
    testScores_mae = [(np.mean(np.abs(testY[grid,:].flatten()- testPredict[grid,:].flatten()))) for grid,_ in enumerate(testPredict)]
    train_all_mae = np.mean(np.abs(trainY.flatten()-trainPredict.flatten()))
    test_all_mae = np.mean(np.abs(testY.flatten()-testPredict.flatten()))
    
    print('grid train nrmse mean:',np.mean(trainScores))
    print('grid train nrmse std:',np.std(trainScores))
    #print('test scores:', len(testScores), testScores)
    print('grid test nrmse mean:',np.mean(testScores))
    print('grid test nrmse std:',np.std(testScores))
    
    print('#'*10)
    print(modelbase,'  ','pred:',predstep)
    print('grid train rmse mean:',np.mean(trainScores_rmse))
    print('grid train rmse std:',np.std(trainScores_rmse))
    print('grid test rmse mean:',np.mean(testScores_rmse))
    print('grid test rmse std:',np.std(testScores_rmse))
    
    print('#'*10)
    print('train_all_mae:',train_all_mae)
    print('test_all_mae:',test_all_mae)
    print('grid train mae mean:',np.mean(trainScores_mae))
    print('grid train mae std:',np.std(trainScores_mae))
    print('grid test mae mean:',np.mean(testScores_mae))
    print('grid test mae std:',np.std(testScores_mae))
        
if __name__ == '__main__':
    main()




    
         
