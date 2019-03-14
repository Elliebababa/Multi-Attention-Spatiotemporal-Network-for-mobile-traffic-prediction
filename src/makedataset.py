import pandas as pd
import numpy as np
import h5py as h5
from utils import *
import pickle
from sklearn.preprocessing import MinMaxScaler

obs_len = 6
pre_len = 1
data_file_path = '../data/processed/Nov_internet_data_t10_s3030_4070.h5'
cl = 0#144    #cyclic len
sl = 0#144*7  #seasonal len
test_len = 144 * 7

def makedataset(data, lookback = obs_len, prestep = pre_len, cl = cl, sl = sl, history = False):
    #make dataset for seq2seq model, includes encoder input data, decoder input aux, decoder target data
    #input data shape: [slots, features, nodes]
    #output shape:
    #   encoder_input_data: [nodes, samples, obs_len, features]
    #   decoder_input_aux: [nodes, samples, pre_len, 2(cl and sl), features2]
    #   decoder_target_data: [nodes, samples, pre_len, 1]
    
    #first we need to scale the data
    #mmn = MinMaxNormalization()
    #data = mmn.fit_transform(data) 
    
    encoder_input_data = []
    decoder_target_data = []
    decoder_input_his = []
    data = np.asarray(data)
    assert np.ndim(data) == 3
    slots, features, nodes = data.shape
    data = np.swapaxes(data, 0, 2) #[nodes, features, slots]
    data = np.swapaxes(data, 1, 2) #[nodes, slots, features]
    #I will skip the first seasonal length for comparison with the historical input
    for n in data[:]:
        input_data = []
        target_data = []
        input_his = []
        for i in range(sl, len(n)-lookback-prestep + 1):
            input_data.append(n[i:i + lookback,])
            target_data.append(n[i + lookback:i + lookback + prestep,0,])
            input_his.append([n[i - cl,],n[i - sl,]])
        '''output for debugging
        print(n[sl:sl+10])
        print(input_data[:2])
        print(target_data[:2])
        print(input_his[:2])
        print(n[:10])
        '''
        encoder_input_data.append(input_data)
        decoder_target_data.append(target_data)
        decoder_input_his.append(input_his)
        '''
        print('finish loading node%d'%len(encoder_input_data))
        '''
    encoder_input_data = np.asarray(encoder_input_data)
    decoder_target_data = np.asarray(decoder_target_data)
    decoder_input_his = np.asarray(decoder_input_his)
    return encoder_input_data, decoder_target_data, decoder_input_his

def splitdata(data, testlen = test_len, shuffle = True):
    #split data into train and test set
    #data is a list of data(encoder,decoder,etc..), and each should has the same length
    #expect each data in datalist should be in the shape of [nodes, samples, obs_len, ..]
    #by default, we will use 7 days of the data for testing
    encoder_input_data, decoder_target_data, decoder_input_his,encoder_input_aux = data
    train_encoder_input, train_decoder_target, train_decoder_input_his = encoder_input_data[:,:-testlen,], decoder_target_data[:,:-testlen,], decoder_input_his[:,:-testlen,]
    test_encoder_input, test_decoder_target, test_decoder_input_his = encoder_input_data[:,-testlen:,], decoder_target_data[:,-testlen:,], decoder_input_his[:,-testlen:,]
    train_encoder_input_aux,test_encoder_input_aux = encoder_input_aux[:,:-testlen,],encoder_input_aux[:,-testlen:,]
    train_decoder_input = np.zeros(train_decoder_target.shape)
    train_decoder_input[:,:,1:,] = train_decoder_target[:,:,:-1]
    train_decoder_input[:,:,0,] = train_encoder_input[:,:,-1,0]
    if (len(train_decoder_input.shape) == 3):
        train_decoder_input = np.expand_dims(train_decoder_input,axis = -1)
    if (len(train_decoder_target.shape) == 3):
        train_decoder_target = np.expand_dims(train_decoder_target, axis = -1)
    if (len(test_decoder_target.shape) == 3):
        test_decoder_target = np.expand_dims(test_decoder_target, axis = -1)

    return train_encoder_input_aux,test_encoder_input_aux,train_encoder_input, train_decoder_target, train_decoder_input_his, test_encoder_input, test_decoder_target, test_decoder_input_his, train_decoder_input
    


if __name__ == '__main__':
   
    f = h5.File(data_file_path,'r')
    data = f['data']
    s = data.shape
    data = np.reshape(data,(s[0],s[1],s[2]*s[3]))
    t1,t2,t3 = makedataset(data,history = False)
    np.save('encoder_input_data.npy',t1)
    np.save('decoder_target_data.npy',t2)
    np.save('decoder_input_his.npy',t3)
    
    
    f2 = h5.File('../data/processed/Nov_call_data_t10_s3030_4070.h5','r')
    f3 = h5.File('../data/processed/Nov_sms_data_t10_s3030_4070.h5','r')
    data2 = f2['data'].value
    data3 = f3['data'].value

    s2 = data2.shape
    data2 = np.reshape(data2,(s2[0],s2[1],s2[2]*s2[3]))
    s3 = data3.shape
    data3 = np.reshape(data2,(s3[0],s3[1],s3[2]*s3[3]))
    data23 = np.concatenate([data2,data3],axis = 1)
    aux, _ , _= makedataset(data23,history = False) 
    print('data_aux',data23.shape)
    print('aux',aux.shape)
    np.save('encoder_input_aux.npy',aux)
 

    t1 = np.load('encoder_input_data.npy','r')
    t2 = np.load('decoder_target_data.npy','r')
    t3 = np.load('decoder_input_his.npy','r')
    t4 = np.load('encoder_input_aux.npy','r')
    daux,daux2,d1,d2,d3,d4,d5,d6,d7 = splitdata([t1,t2,t3,t4])
    #return train_encoder_input_aux,test_encoder_input_aux,train_encoder_input, train_decoder_target, train_decoder_input_his, test_encoder_input, test_decoder_target, test_decoder_input_his, train_decoder_input
    
    f = h5.File('train_test_set_6_1_Nov_2.h5','w')
    f.create_dataset('description', data = 'data with none historical data')
    f.create_dataset('train_encoder_input_aux',data = daux)
    f.create_dataset('test_encoder_input_aux',data = daux2)
    f.create_dataset('train_encoder_input',data = d1)
    f.create_dataset('train_decoder_target',data = d2)
    #f.create_dataset('train_decoder_input_his',data = d3 )
    f.create_dataset('test_encoder_input',data = d4)
    f.create_dataset('test_decoder_target',data = d5)
    #f.create_dataset('test_decoder_input_his',data = d6)
    f.create_dataset('train_decoder_input',data = d7)
    f.close()
    




