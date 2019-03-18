import pandas as pd
import numpy as np
import h5py as h5
import time
from threading import Thread,Lock
from joblib import Parallel, delayed
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings("ignore")

intervals = 144
daytest = 7
testlen = intervals*daytest
stepahead = 1
#f1 = h5.File('../../../data/processed/internet_t10_s3030_4070.h5','r')
f1 = h5.File('../data/processed/Nov_internet_data_t10_s3030_4070.h5','r')
dataX = f1['data'].value
dataX = np.squeeze(dataX)
dataX = np.reshape(dataX,(-1,900))
dataX = dataX[:].T[:]
#cfg = (2,2,3)
print('dataX.shape: ',dataX.shape)
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

def walk_forward_validation(X, config, stepahead = 1):
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

def test_set_validation(X, config, stepahead = 1):
    #get the resid for all to make more accurate predictions
    m = ARIMA(X, order = config)
    mf = m.fit(transparams = True,trend = 'nc', disp = False)
    #this part is to validate the test set without rolling mechanism
    predictions = list()
    trainx, testx = X[:-testlen], X[-testlen:]
    history = [x for x in trainx]
    predictions = list()
    model = ARIMA(history, order = config)
    model_fit = model.fit(transparams = True,trend = 'nc', disp = False)
    ar_coef = mf.arparams
    ma_coef = mf.maparams
    for t in range(len(testx)):
        diff = difference(history)
        resid = mf.resid[:-testlen+t]
        yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
        predictions.append(yhat)
        obs = testx[t]
        history.append(obs)
    predictions = np.asarray(predictions)
    nrmse_score = Nor_rmse(testx, predictions)
    all_true_value.append(testx)
    all_predict_value.append(predictions)
    return nrmse_score


def score_model(data, cfg, stepahead = 1, debug = False):
    gridNum,X = data
    result = None
    if debug:
        result = test_set_validation (X,cfg,stepahead) # test_set_validation,  walk_forward_validation 
    else:
        try:
            result = test_set_validation (X,cfg,stepahead)# test_set_validation,  walk_forward_validation 
        except:
            result = None
    if result is not None:# and gridNum % 100 == 0:
            print('gridNum:%d, nrmse:%.8f'%(gridNum,result))
    return(gridNum, result)


def arima_test_all(dataX, cfg, stepahead = 1,parallel = False, n_jobs = 32):
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
            nrmse = score_model(data,cfg,stepahead)
            scores.append(nrmse)
    scores = [r for r in scores if r[1]!= None]
    print('scores:',scores)
    nrmses = list(list(zip(*scores))[1])
    print('finish all...')
    print('mean nrmses: %.8f'%(np.mean(nrmses)))
    print('std nrmses: %.8f'%(np.std(nrmses)))
    print('all nrmse: %8f'%(Nor_rmse(all_true_value,all_predict_value)))
    return scores

if __name__ == '__main__':
    print('ARIMA TESTING...')
    tss = time.time()
    print('dataX, shape = ',dataX.shape,'\n')
    arima_test_all(dataX,cfg,stepahead)
    print('finish all time:%.3f'%(time.time()-tss))

