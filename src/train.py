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
#keras
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
#utils and metrics for training
from utils import *
import metrics
#model
from models import *
#define parameters for training 
days_test = 7
T = 144
len_test = T * days_test
lr = 0.0002
look_back = 6 # look back of interval, the same as len_closeness
nb_epoch = 200
patience = 10  # early stopping patience
nb_epoch_cont = 0 #epoch for continue training 
batch_size = 2**10
verbose = 2
#model for training
modelbase = 'seq2seq'

print('\n','='*5, ' configuration ', '='*5)
print('\nlr: %.5f, lookback: %d, nb_epoch: %d, patience:%d, nb_epoch_cont: %d, batch size:%d'%(lr, look_back, nb_epoch, patience, nb_epoch_cont, batch_size))
print('\n','='*15,'\n')

def display_evaluation(y_pred, y_true, info = 'evaluation:'):
    assert(np.shape(y_pred) == np.shape(y_true))
    score_mse = np.mean(np.power(y_pred-y_true,2))
    score_rmse = np.power(score_mse, 0.5)
    score_nrmse = score_rmse / np.mean(y_true)
    print('{} mse:{}, rmse:{}, nrmse:{}'.format(info,score_mse, score_rmse, score_nrmse))
    
def build_model(modelbase = modelbase):
    if modelbase == 'seq2seq':
        model, encoder_model, decoder_model = seq2seq()
    
    adam = Adam(lr = lr)
    model.compile(loss = 'mse', optimizer = adam, metrics = [metrics.rmse, metrics.mape, metrics.ma])
    model.summary()

    return model, encoder_model, decoder_model


import click
@click.command()
@click.option('--modelbase',default= modelbase, help='seq2seq')
def main(modelbase):
    modelbase = modelbase
    # load data and make dataset
    print('loading data...')
    ts = time.time()
    train_encoder_input, train_encoder_input_aux, train_decoder_input, train_decoder_input_his, train_decoder_target, test_encoder_input, test_encoder_input_aux, test_decoder_input_his, test_decoder_target, mmn = loadData()

    print('Train set shape: \ntrain_encoder_input:{}, \ntrain_encoder_input_aux:{}, \ntrain_decoder_input:{}, \ntrain_decoder_input_his:{}, \ntrain_decoder_target:{}'.format(
       train_encoder_input.shape, train_encoder_input_aux.shape, train_decoder_input.shape, train_decoder_input_his.shape, train_decoder_target.shape ))
    print('Test set shape: \ntest_encoder_input:{}, \ntest_encoder_input_aux:{}, \ntest_decoder_input_his:{}, \ntest_decoder_target:{}'.format(
        test_encoder_input.shape, test_encoder_input_aux.shape, test_decoder_input_his.shape, test_decoder_target.shape ))


    print('\n\nstacking data...')
    train_encoder_input = np.vstack(train_encoder_input)
    train_encoder_input_aux = np.vstack(train_encoder_input_aux)
    train_decoder_input = np.vstack(train_decoder_input)
    train_decoder_input_his = np.vstack(train_decoder_input_his)
    train_decoder_target = np.vstack(train_decoder_target)
    test_encoder_input = np.vstack(test_encoder_input)
    test_encoder_input_aux = np.vstack(test_encoder_input_aux)
    test_decoder_input_his = np.vstack(test_decoder_input_his)
    test_decoder_target = np.vstack(test_decoder_target)
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
    model.save_weights(fname_param)
    print("\ntrain model elapsed time (training): %.3f seconds\n" % (time.time() - ts))
    
    #==== evaluating model ===================================================================================

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(fname_param)
    #train score
    train_pred =  model.predict([train_encoder_input, train_decoder_input], verbose = verbose, batch_size = batch_size)
    train_pred_orig = mmn.inverse_transform(train_pred)
    train_decoder_target_orig = mmn.inverse_transform(train_decoder_target)
    display_evaluation(train_pred_orig,train_decoder_target_orig,'Train score')

    #test score
    test_pred = decoder_prediction([test_encoder_input], encoder_model, decoder_model)
    test_pred_orig = mmn.inverse_transform(test_pred)
    test_decoder_target_orig = mmn.inverse_transform(test_decoder_target)
    display_evaluation(test_pred_orig,test_decoder_target_orig,'Test score')
    print("\nevaluate model elapsed time (eval): %.3f seconds\n" % (time.time() - ts))
    
    #==== continue to train model ==============================================================================
    
    print('=' * 10)
    print("training model (cont)...")
    ts = time.time()
    fname_param = os.path.join('{}.cont.best.h5'.format(modelbase))
    model_checkpoint = ModelCheckpoint(fname_param, monitor='rmse', verbose=verbose, save_best_only=True, mode='min')
    tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 0, write_graph = True, write_images = False, embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)
    if modelbase =='seq2seq':
        history = model.fit([train_encoder_input, train_decoder_input], train_decoder_target, verbose = verbose, batch_size = batch_size, epochs = nb_epoch_cont, validation_split = 0.2)     
    model.save_weights(os.path.join('{}_cont.h5'.format(modelbase)), overwrite=True)
    print("\ncont train model elapsed time (training cont): %.3f seconds\n" % (time.time() - ts))
    
    #==== evaluate on the final model ===============================================================================
    
    print('=' * 10)
    print('evaluating using the final model')
    model.load_weights(os.path.join('{}_cont.h5'.format(modelbase)))
    #train score
    train_pred =  model.predict([train_encoder_input, train_decoder_input], verbose = verbose, batch_size = batch_size)
    train_pred_orig = mmn.inverse_transform(train_pred)
    train_decoder_target_orig = mmn.inverse_transform(train_decoder_target)
    display_evaluation(train_pred_orig,train_decoder_target_orig,'Train score')
    #test score
    test_pred = decoder_prediction([test_encoder_input], encoder_model, decoder_model)
    test_pred_orig = mmn.inverse_transform(test_pred)
    test_decoder_target_orig = mmn.inverse_transform(test_decoder_target)
    display_evaluation(test_pred_orig,test_decoder_target_orig,'Test score')

    print("\nevaluate final model elapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))
    
if __name__ == '__main__':
    main()




    
         
