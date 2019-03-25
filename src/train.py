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
days_test = 7
T = 144
len_test = T * days_test
lr = 0.0002
look_back = 6 # look back of interval, the same as len_closeness
nb_epoch = 100
nb_epoch_cont = 0
patience = 5  # early stopping patience
batch_size = 2**10
verbose = 2
#model for training
modelbase = 'MAModel-global' # lstm, seq2seq, MAModel, MAModel-global
m = 64 #hidden layer of MAModel
predstep = 1

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
    elif modelbase == 'lstm':
        model = lstm(predstep = predstep) # predstep = 1
        encoder_model = None
        decoder_model = model
    elif modelbase == 'MAModel':
        a = MAModel()
        model = a.build_model(input_dim = int(5))
        encoder_model = None
        decoder_model = None
    elif modelbase == 'MAModel-global':
        a = MAModel(global_att = True)
        model = a.build_model(input_dim = int(5))
        encoder_model = None
        decoder_model = None
    adam = Adam(lr = lr)
    model.compile(loss = 'mse', optimizer = adam, metrics = [metrics.rmse, metrics.mape, metrics.ma])
    model.summary()
    
    return model, encoder_model, decoder_model

import click
@click.command()
@click.option('--modelbase',default= modelbase, help='lstm, seq2seq, MAModel, MAModel-global')
def main(modelbase):
    modelbase = modelbase
    # load data and make dataset
    print('loading data...')
    ts = time.time()
    fname = 'train_test_set_en6_de{}_Nov_neighbor.h5'.format(predstep)
    if modelbase == 'MAModel-global':
        fname = 'train_test_set_en6_de1_Nov_neighbor.h5'
    train_encoder_input, train_encoder_input_aux, train_decoder_input, train_decoder_input_his, train_decoder_target, test_encoder_input, test_encoder_input_aux, test_decoder_input_his, test_decoder_target, \
        train_neighbor_values, test_neighbor_values, train_neighbor_weights, test_neighbor_weights = loadData(fname)

    print('Train set shape: \ntrain_encoder_input:{}, \ntrain_encoder_input_aux:{}, \ntrain_decoder_input:{}, \ntrain_decoder_input_his:{}, \ntrain_decoder_target:{}'.format(
       train_encoder_input.shape, train_encoder_input_aux.shape, train_decoder_input.shape, train_decoder_input_his.shape, train_decoder_target.shape ))
    print('Test set shape: \ntest_encoder_input:{}, \ntest_encoder_input_aux:{}, \ntest_decoder_input_his:{}, \ntest_decoder_target:{}'.format(
        test_encoder_input.shape, test_encoder_input_aux.shape, test_decoder_input_his.shape, test_decoder_target.shape ))

    gridNum, trainSlots, lookback, _ = train_encoder_input.shape
    _, _, predSteps, _ = train_decoder_target.shape
    _, testSlots, _, _ = test_encoder_input.shape
    
    #test decoder input
    test_decoder_input = np.zeros(test_decoder_target.shape)
    test_decoder_input[:,:,1:,] = test_decoder_target[:,:,:-1]
    test_decoder_input[:,:,0,] = test_encoder_input[:,:,-1,0:1]

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
    if modelbase in ['MAModel','MAModel-global']:
        train_encoder_input = np.concatenate([train_encoder_input, train_encoder_input_aux],axis = -1)
        test_encoder_input = np.concatenate([test_encoder_input, test_encoder_input_aux],axis = -1)
        
    #neighbor_weights and neighbor_values
    if modelbase == 'MAModel-global':
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
    
    #decoder_input_his
    train_decoder_input_his = np.vstack(np.swapaxes(train_decoder_input_his,0,1))
    test_decoder_input_his = np.vstack(np.swapaxes(test_decoder_input_his,0,1))
    
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
    fname_param = os.path.join('{}_pred{}.best.h5'.format(modelbase, predstep))
    #callbacks
    early_stopping = EarlyStopping(monitor='val_rmse', patience=patience, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
    tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 0, write_graph = True, write_images = False, embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)
    print("\ncompile model elapsed time (compiling model): %.3f seconds\n" %(time.time() - ts))
    
    #==== training model ===================================================================================
    
    print('=' * 10)
    if nb_epoch > 0:
    #train model when nb_epoch is set larger than 0
        print("training model...")
        ts = time.time()
        #history = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, model_checkpoint,tensor_board], verbose=verbose)
        callbacks = [early_stopping, model_checkpoint,tensor_board]
        if modelbase =='seq2seq':
            history = model.fit([train_encoder_input, train_decoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.2, callbacks = callbacks)
        elif modelbase =='lstm':
            history = model.fit([train_encoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.2, callbacks = callbacks)
        elif modelbase =='MAModel':
            s0_train = h0_train = np.zeros((train_encoder_input.shape[0],m))
            enc_att = np.ones((train_encoder_input.shape[0],1,train_encoder_input.shape[2]))
            history = model.fit([train_encoder_input,h0_train,s0_train,enc_att,train_decoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.2, callbacks = callbacks)
        elif modelbase =='MAModel-global':
        #model = Model([encoder_inputs_local, encoder_inputs_global_value,encoder_inputs_global_weight, h0, s0, enc_att_local, enc_att_global, decoder_inputs], output)
            s0_train = h0_train = np.zeros((train_encoder_input.shape[0],m))
            enc_att = np.ones((train_encoder_input.shape[0],1,train_encoder_input.shape[2]))
            enc_att_glo = np.ones((train_neighbor_values.shape[0],1,train_neighbor_values.shape[2]))
            history = model.fit([train_encoder_input, train_neighbor_values, train_neighbor_weights, h0_train, s0_train, enc_att, enc_att_glo, train_decoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.2, callbacks = callbacks)
        
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
    elif modelbase == 'lstm':
        train_pred =  model.predict([train_encoder_input], verbose = verbose, batch_size = batch_size)
    elif modelbase =='MAModel':
        s0_train = h0_train = np.zeros((train_encoder_input.shape[0],m))
        enc_att = np.ones((train_encoder_input.shape[0],1,train_encoder_input.shape[2]))
        train_pred =  model.predict([train_encoder_input,h0_train,s0_train,enc_att, train_decoder_input], verbose = verbose, batch_size = batch_size)
    elif modelbase =='MAModel-global':
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
    if modelbase =='seq2seq':
        test_pred = decoder_prediction([test_encoder_input], encoder_model, decoder_model, pre_step = predstep)
        #test_pred = model.predict([test_encoder_input, test_decoder_input], encoder_model, decoder_model)
    elif modelbase == 'lstm':
        test_pred = model.predict([test_encoder_input], verbose = verbose, batch_size = batch_size)
    elif modelbase =='MAModel':
        s0_test = h0_test = np.zeros((test_encoder_input.shape[0],m))
        enc_att = np.ones((test_encoder_input.shape[0],1,test_encoder_input.shape[2]))
        test_pred = model.predict([test_encoder_input,h0_train,s0_train,enc_att, test_decoder_input], verbose = verbose, batch_size = batch_size)
    elif modelbase =='MAModel-global':
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

    #first we need to unstack data
    '''
    print('unstacking data into seperate training grids...')
    train_encoder_input = np.reshape(train_encoder_input, (gridNum, trainSlots, lookback, -1))
    test_encoder_input = np.reshape(test_encoder_input,(gridNum, testSlots, lookback, -1))
    
    train_encoder_input_aux = np.reshape(train_encoder_input_aux, (gridNum, trainSlots, lookback, -1))
    test_encoder_input_aux = np.reshape(test_encoder_input_aux, (gridNum, testSlots, lookback, -1))
            
    #neighbor_weights and neighbor_values
    if modelbase == 'MAModel-global':
        train_neighbor_values = np.reshape(train_neighbor_values, (gridNum, trainSlots, lookback, -1))
        test_neighbor_values = np.reshape(test_neighbor_values, (gridNum, testSlots, lookback, -1))
        train_neighbor_weights = np.reshape(train_neighbor_weights, (gridNum, trainSlots, lookback, -1))
        test_neighbor_weights = np.reshape(test_neighbor_weights, (gridNum, testSlots, lookback, -1))
        print('train_neighbor_values:{}, \ntest_neighbor_values:{}'.format(train_neighbor_values.shape, test_neighbor_values.shape))
        print('train_neighbor_weights:{}, \ntest_neighbor_weights:{}'.format(train_neighbor_weights.shape, test_neighbor_weights.shape))
    
    #decoder input
    train_decoder_input = np.reshape(train_decoder_input, (gridNum, trainSlots, predstep, -1))
    test_decoder_input = np.reshape(test_decoder_input, (gridNum, testSlots, 1, -1))
    
    #decoder_input_his
    #train_decoder_input_his = np.reshape(train_decoder_input_his, (gridNum, trainSlots, 1, -1))
    #test_decoder_input_his = np.reshape(test_decoder_input_his, (gridNum, trainSlots, 1, -1))
    
    #decoder_target
    train_decoder_target = np.reshape(train_decoder_target, (gridNum, trainSlots, 1, -1))
    test_decoder_target = np.reshape(test_decoder_target, (gridNum, testSlots, 1, -1))    

    
    #train score
    ind = 0
    trainScores = []
    for tra_enc_inp, tra_dec_inp,tra_dec_tar  in zip(train_encoder_input, train_decoder_input,train_decoder_target):
        if modelbase =='seq2seq':
            train_pred =  model.predict([tra_enc_inp, tra_dec_inp], verbose = 0, batch_size = batch_size)
        elif modelbase == 'lstm':
            train_pred =  model.predict([tra_enc_inp], verbose = 0, batch_size = batch_size)
        elif modelbase =='MAModel':
            s0_train = h0_train = np.zeros((tra_enc_inp.shape[0],m))
            enc_att = np.ones((tra_enc_inp.shape[0],1,tra_enc_inp.shape[2]))
            train_pred =  model.predict([tra_enc_inp,h0_train,s0_train,enc_att, tra_dec_inp], verbose = 0, batch_size = batch_size)
        elif modelbase =='MAModel-global':
        #model = Model([encoder_inputs_local, encoder_inputs_global_value,encoder_inputs_global_weight, h0, s0, enc_att_local, enc_att_global, decoder_inputs], output)
            s0_train = h0_train = np.zeros((tra_enc_inp.shape[0],m))
            enc_att = np.ones((tra_enc_inp.shape[0],1,tra_enc_inp.shape[2]))
            enc_att_glo = np.ones((train_neighbor_values[ind].shape[0],1,train_neighbor_values[ind].shape[2]))
            train_pred=model.predict([tra_enc_inp, train_neighbor_values[ind], train_neighbor_weights[ind], h0_train, s0_train, enc_att, enc_att_glo, tra_dec_inp],verbose = 0, batch_size=batch_size)


        train_pred_orig = mmn.inverse_transform(train_pred.reshape(train_pred.shape[0],-1))
        train_decoder_target_orig = mmn.inverse_transform(tra_dec_tar.reshape(tra_dec_tar.shape[0],-1))
        score_mse = np.mean(np.power(train_pred_orig-train_decoder_target_orig,2))
        score_rmse = np.power(score_mse, 0.5)
        train_nrmse = score_rmse / np.mean(train_decoder_target_orig)
        trainScores.append(train_nrmse)
        ind += 1

    #test score
    ind = 0
    testScores = []
    for tes_enc_inp, tes_dec_inp, tes_dec_tar in zip(test_encoder_input, test_decoder_input, test_decoder_target):
        if modelbase =='seq2seq':
            test_pred = decoder_prediction([tes_enc_inp], encoder_model, decoder_model, pre_step = predstep)
        elif modelbase == 'lstm':
            test_pred = model.predict([tes_enc_inp], verbose = 0, batch_size = batch_size)
        elif modelbase =='MAModel':
            s0_test = h0_test = np.zeros((tes_enc_inp.shape[0],m))
            enc_att = np.ones((test_encoder_input.shape[0],1,tes_enc_inp.shape[2]))
            test_pred = model.predict([tes_enc_inp,h0_train,s0_train,enc_att, tes_dec_inp], verbose = 0, batch_size = batch_size)
        elif modelbase =='MAModel-global':
            s0_test = h0_test = np.zeros((tes_enc_inp.shape[0],m))
            enc_att = np.ones((tes_enc_inp.shape[0],1,tes_enc_inp.shape[2]))
            enc_att_glo = np.ones((test_neighbor_values[ind].shape[0],1,test_neighbor_values[ind].shape[2]))
            test_pred = model.predict([tes_enc_inp, test_neighbor_values[ind], test_neighbor_weights[ind], h0_test, s0_test, enc_att, enc_att_glo, tes_dec_inp], verbose = 0, batch_size = batch_size)
        
        
        test_pred_orig = mmn.inverse_transform(test_pred.reshape(test_pred.shape[0],-1))
        test_decoder_target_orig = mmn.inverse_transform(tes_dec_tar.reshape(tes_dec_tar.shape[0],-1))
        score_mse = np.mean(np.power(train_pred_orig-train_decoder_target_orig,2))
        score_rmse = np.power(score_mse, 0.5)
        test_nrmse = score_rmse / np.mean(test_decoder_target_orig)
        testScores.append(test_nrmse)
        ind += 1
    '''
    
    train_pred_orig = np.reshape(train_pred_orig,(trainSlots,gridNum,1))
    train_decoder_target_orig = np.reshape(train_decoder_target_orig,(trainSlots,gridNum,1))
    
    test_pred_orig = np.reshape(test_pred_orig,(testSlots,gridNum,1))
    test_decoder_target_orig = np.reshape(test_decoder_target_orig,(testSlots,gridNum,1))
    

    trainPredict = np.swapaxes(train_pred_orig,0,1)
    testPredict = np.swapaxes(test_pred_orig,0,1)

    trainY = np.swapaxes(train_decoder_target_orig, 0, 1)
    testY = np.swapaxes(test_decoder_target_orig, 0, 1)

    print('trainY shape:',trainY.shape)
    trainScores = [(math.sqrt(mean_squared_error(trainY[grid,:], trainPredict[grid,:, 0:1])))/np.mean(trainY[grid,:]) for grid,_ in enumerate(trainPredict)]
    testScores = [(math.sqrt(mean_squared_error(testY[grid,:], testPredict[grid,:, 0:1])))/np.mean(testY[grid,:]) for grid,_ in enumerate(testPredict)]
    
    
    print('grid train nrmse mean:',np.mean(trainScores))
    print('grid train nrmse std:',np.std(trainScores))
    #print('test scores:', len(testScores), testScores)
    print('grid test nrmse mean:',np.mean(testScores))
    print('grid test nrmse std:',np.std(testScores))


    
    
if __name__ == '__main__':
    main()




    
         
