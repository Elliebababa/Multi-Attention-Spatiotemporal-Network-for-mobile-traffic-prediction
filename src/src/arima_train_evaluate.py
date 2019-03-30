import pandas as pd
import numpy as np
import h5py as h5
import time
from threading import Thread,Lock
from joblib import Parallel, delayed
from statsmodels.tsa.arima_model import ARIMA
import warnings
import sys
warnings.filterwarnings("ignore")

intervals = 144
daytest = 7
testlen = intervals*daytest
stepahead = 1
#f1 = h5.File('../../../data/processed/internet_t10_s3030_4070.h5','r')
f1 = h5.File('../../data/processed/Nov_internet_data_t10_s3030_4070.h5','r')
dataX = f1['data'].value
dataX = np.squeeze(dataX)
dataX = np.reshape(dataX,(-1,900))
dataX = dataX[:].T[:]
#cfg = (2,2,3)
print('dataX.shape: ',dataX.shape)
#testlen = 1008#int(len(dataX[0,:])/3)
cfg = (3,1,2)#(p,d,q)
lock = Lock()
all_true_value = []
all_predict_value = []

def Nor_rmse(testx, predictions):
    #notice that testx is array while predictions is list of array
    #we should first convert list of array into array
    if type(testx) == list:
        testx = np.asarray(testx).reshape(-1)
    if type(predictions) == list:
        predictions = np.asarray(predictions).reshape(-1)
    assert predictions.shape == testx.shape
    meant = np.mean(testx)
    mse = np.mean(np.power(testx-predictions,2))
    rmse = np.sqrt(mse)
    nrmse = rmse/meant
    return nrmse
'''
def mse(testx, predictions):
    #notice that testx is array while predictions is list of array
    #we should first convert list of array into array
    if type(testx) == list:
        testx = np.asarray(testx).reshape(-1)
    if type(predictions) == list:
        predictions = np.asarray(predictions).reshape(-1)
    assert predictions.shape == testx.shape
    meant = np.mean(testx)
    mse = np.mean(np.power(testx-predictions,2))
    return mse
'''
def cal_mae(testx, predictions):
    if type(testx) == list:
        testx = np.asarray(testx).reshape(-1)
    if type(predictions) == list:
        predictions = np.asarray(predictions).reshape(-1)
    assert predictions.shape == testx.shape
    mae = np.mean(np.abs(testx-predictions))
    return mae
    
def cal_rmse(testx, predictions):
    if type(testx) == list:
        testx = np.asarray(testx).reshape(-1)
    if type(predictions) == list:
        predictions = np.asarray(predictions).reshape(-1)
    assert predictions.shape == testx.shape
    mse = np.mean(np.power(testx-predictions,2))
    rmse = np.sqrt(mse)
    return rmse    

def walk_forward_validation(X, config, stepahead = 3):
    predictions = list()
    trainx, testx = X[:-testlen], X[-testlen:]
    history = [x for x in trainx]
    ts = time.time()
    for i in range(0,len(testx),stepahead):
        for step in range(1,stepahead+1):
            if i%500 == 0:
                print('i:%d, step:%d, time:%.3f'%(i,step,time.time()-ts))
            model = ARIMA(history, order = config)
            model_fit = model.fit(transparams = True,disp=0,trend = 'nc',)
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(yhat)
            ar_coef = model_fit.arparams
            ma_coef = model_fit.maparams
            resid = model_fit.resid
        history[-stepahead:] = testx[i:i+stepahead]
    nrmse_score = Nor_rmse(testx, predictions)
    #global all_true_value
    #global all_predict_value
    #lock.acquire()
    all_true_value.append(testx)
    all_predict_value.append(predictions)
    #lock.release()
    return nrmse_score

def predict(coef, history):
    #predict function for manual prediction
    yhat = 0.0
    for i in range(1, len(coef)+1):
        yhat += coef[i - 1]*history[-i]
    return yhat

def difference(dataset):
    #making manual predictions for the arima model calculate the difference first
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)

def inv_dif(s,dataset):
    res = [s+dataset[0]]
    for i in range(1,len(dataset)):
        res.append(dataset[i] + res[i-1])
    return res
    

def insample_validation___(X, config, stepahead):
    #get the resid for all to make more accurate predictions
    m = ARIMA(X, order = config)
    mf = m.fit(transparams = 0,trend = 'c', disp = False)
    #this part is to validate the test set without rolling mechanism
    trainx, testx = X[:-testlen], X[-testlen:]
    st = len(trainx)-1
    predictions = mf.predict(start = st, end = st+len(testx)-1, dynamic = False)
    predictions2 = inv_dif(X[st], predictions)
    print(predictions2[:5])
    print(testx[:5])
    testvalues = testx
    predictions = np.asarray(predictions)
    testvalues = np.asarray(testvalues)
    nrmse_score = Nor_rmse(testvalues, predictions)
    mae_score = cal_mae(testvalues, predictions)
    rmse_score = cal_rmse(testvalues, predictions)
    all_true_value.append(testvalues)
    all_predict_value.append(predictions)
    return nrmse_score,mae_score,rmse_score


def test_set_validation(X, config, stepahead):
    #get the resid for all to make more accurate predictions
    m = ARIMA(X, order = config)
    mf = m.fit(transparams = True,trend = 'nc', disp = False)
    #this part is to validate the test set without rolling mechanism
    trainx, testx = X[:-testlen], X[-testlen:]
    history = [x for x in trainx]
    predictions = list()
    testvalues = list()
    #model = ARIMA(history, order = config)
    #model_fit = model.fit(transparams = True,trend = 'nc', disp = False)
    ar_coef = mf.arparams
    ma_coef = mf.maparams
    for t in range(len(testx)-stepahead+1):
        #print('\nT ',t)
        #if t == 3:
        #    sys.exit()
        pred = []
        for step in range(1,stepahead+1):
            diff = difference(history)
            resid = mf.resid[:len(trainx)+t-1]
            #print(len(resid))
            #tr_coef = model_fit.params[0]
            #mu = tr_coef*(1 - ar_coef.sum())
            yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)# + mu
            #yhat = history[len(trainx)+step+t-2] + predict(ar_coef, diff) + predict(ma_coef, resid)# + mu
            #print('yhat: ',yhat,'  fit:',model_fit.forecast()[0], yhat == model_fit.forecast()[0])
            pred.append(yhat)
            #print('    step', step)
            #print('    history',history[len(trainx)+step+t-2])
            history.append(yhat)
            #print('    yhat',yhat)
        #print('\npred:' , pred)
        #print('true:',testx[t:t+stepahead])
        predictions.append(pred)
        testvalues.append(testx[t:t+stepahead])
        #history[-stepahead:] = testx[t:t+stepahead]
        history = history[:len(trainx)+t]
        history.append(testx[t])
        #print('now his: ',history[-5:])
    
    
    np.save('arima_predictins_step{}.npy'.format(stepahead), predictions)
    np.save('arima_tests_step{}.npy'.format(stepahead), testvalues)
    
    predictions = np.asarray(predictions)
    testvalues = np.asarray(testvalues)
    #print(predictions[:10])
    #print(testvalues[:10])
    nrmse_score = Nor_rmse(testvalues, predictions)
    mae_score = cal_mae(testvalues, predictions)
    rmse_score = cal_rmse(testvalues, predictions)
    all_true_value.append(testvalues)
    all_predict_value.append(predictions)
    return nrmse_score,mae_score,rmse_score


def score_model(data, cfg, stepahead, debug = False):
    gridNum,X = data
    result = None
    if debug:
        result = test_set_validation(X,cfg,stepahead) # test_set_validation,  walk_forward_validation ,insample_validation 
    else:
        try:
            result = test_set_validation(X,cfg,stepahead)# test_set_validation,  walk_forward_validation ,insample_validation 
        except:
            result = None
    if result is not None:# and gridNum % 100 == 0:
            print('gridNum:%d, nrmse:%.8f, mae:%.8f'%(gridNum,result[0],result[1]))
    return(gridNum, result)


def arima_test_all(dataX, cfg, stepahead,parallel = False, n_jobs = 32):
    gridnums = len(dataX)
    print('testing on %d grids, with p = %d, d = %d, q = %d'%(gridnums,cfg[0], cfg[1], cfg[2]))
    print('forecast %d step ahead..., test len %d...'%(stepahead, testlen))
    scores = None
    if parallel:
        print('using parallel..')
        executor = Parallel(n_jobs = n_jobs ,require = 'sharedmem')
        tasks = (delayed(score_model)(data,cfg,stepahead) for data in enumerate(dataX))
        scores = executor(tasks)
    else:
        scores = list()
        for data in enumerate(dataX):
            sco = score_model(data,cfg,stepahead)
            scores.append(sco)
    scores = [r for r in scores if r[1]!= None]
    #print('scores:',scores)
    nrmses = [i[1][0] for i in scores]
    maes = [i[1][1] for i in scores]
    rmses = [i[1][2] for i in scores]
    #print(nrmses)
    print('finish all...')
    print('mean nrmses: %.8f'%(np.mean(nrmses)))
    print('std nrmses: %.8f'%(np.std(nrmses)))
    print('mean maes: %.8f'%(np.mean(maes)))
    print('std maes: %.8f'%(np.std(maes)))
    print('mean rmses: %.8f'%(np.mean(rmses)))
    print('std rmses: %.8f'%(np.std(rmses)))
    print('\n\n')
    print('all nrmse: %8f'%(Nor_rmse(all_true_value,all_predict_value)))
    print('all mae: %8f'%(cal_mae(all_true_value,all_predict_value)))
    print('all rmse: %8f'%(cal_rmse(all_true_value,all_predict_value)))
    
    return scores

if __name__ == '__main__':
    print('ARIMA TESTING...')
    tss = time.time()
    print('dataX, shape = ',dataX.shape,'\n')
    arima_test_all(dataX[:],cfg,stepahead)
    print('finish all time:%.3f'%(time.time()-tss))

