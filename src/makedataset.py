import numpy as np
import pandas as pd
import h5py as h5
import os
from utils import *
import pickle
from sklearn.preprocessing import MinMaxScaler

obs_len = 6
pre_len = 6
genneighbors = True
neighbnum = 15
data_file_path = '../data/processed/Nov_internet_data_t10_s3030_4070.h5'
cl = 0#144    #cyclic len
sl = 0#144*7  #seasonal len
test_len = 144 * 7 - pre_len + 1

genneighbor = True
gensemantic = False
genhis = False
genh5File = True
genaux = True

if genneighbors:
    weights = np.load('./neighbor_weights_matrix_directed.npy')
    neighborsid = np.zeros(shape=(900,neighbnum), dtype = int)
    for i in range(900):
        neighbor = np.argsort(-weights[i,:])[:neighbnum]
        neighborsid[i,:] = neighbor
        #print('neighbor:',weights[i,neighbor])

def makedataset(data, lookback = obs_len, prestep = pre_len, cl = cl, sl = sl, history = False, genneighbors = False, neighnum = 15,gensemantic = False, senmanpath = None):
    #make dataset for seq2seq model, includes encoder input data, decoder input aux, decoder target data
    #if history == True: generate historical values for decoder input
    #if neighbors == True: generate neighbor_values and neigbor_weights for each neighbor
    #input data shape: [slots, features, nodes]
    #output shape:
    #   encoder_input_data: [nodes, samples, obs_len, features]
    #   decoder_input_aux: [nodes, samples, pre_len, 2(cl and sl), features2]
    #   decoder_target_data: [nodes, samples, pre_len, 1]
    #   *neighbor_values:[nodes, samples, obs_len, neighbornum]
    #   *neighbor_weight:[nodes, samples, obs_len, neighbornum]
    #   *semantic_input_data:[nodes, samples, prestep, embedding_dim]
    # first we need to scale the data
    # mmn = MinMaxNormalization()
    # data = mmn.fit_transform(data) 
    
    encoder_input_data = []
    decoder_target_data = []
    decoder_input_his = []
    if genneighbors:
        neighbor_values = []
        neighbor_weight = []
    if gensemantic:
        semantic_input_data = []
    data = np.asarray(data)
    assert np.ndim(data) == 3
    slots, features, nodes = data.shape
    data = np.swapaxes(data, 0, 2) #[nodes, features, slots]
    data = np.swapaxes(data, 1, 2) #[nodes, slots, features]
    #if sl,cl not equal to zero, it will skip the first seasonal length for comparison with the historical input
    for nodeid, n in enumerate(data[:]):
        input_data = []
        target_data = []
        input_his = []
        if genneighbors:
            nei_v = []
            nei_w = []
        if gensemantic:
            semant = []
        for i in range(sl, len(n)-lookback-prestep + 1):
            input_data.append(n[i:i + lookback,])
            target_data.append(n[i + lookback:i + lookback + prestep,0,])
            input_his.append([n[i - cl,],n[i - sl,]])
            if genneighbors:
                #print(neighborsid[nodeid].tolist())
                tmp = [data[int(nid),i:i+lookback,] for nid in neighborsid[nodeid].tolist()]
                tmp = np.transpose(np.squeeze(np.asarray(tmp)))
                #print(tmp.shape)
                nei_v.append(tmp)
                tmp2 = [np.transpose(np.squeeze(weights[nodeid,[neighborsid[nodeid]]]))] * lookback
                #print(tmp2)
                tmp2 = np.asarray(tmp2)
                nei_w.append(tmp2)
                #break
            if gensemantic:
                embedding = np.zeros((10,))
                timestamp = 1383260400000+(i+sl)*10*60000
                emfile = senmanpath + str(timestamp) +'.txt'
                if os.path.exists(emfile):
                    a = pd.read_table(emfile, skiprows = 1, header = None,sep = '\s+',index_col =0)
                    embedding = a.iloc[nodeid+1]
                embedding = [embedding] * prestep
                embedding = np.asarray(embedding)
                semant.append(embedding)
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
        if genneighbors:
            neighbor_values.append(nei_v)
            neighbor_weight.append(nei_w)        
        if gensemantic:
            semantic_input_data.append(semant)
    
    
        '''
        print('finish loading node%d'%len(encoder_input_data))
        '''
    if genneighbors:
        neighbor_values = np.asarray(neighbor_values)
        neighbor_weight = np.asarray(neighbor_weight)
    else:
        neighbor_values = np.zeros((1,1))
        neighbor_weight = np.zeros((1,1))
    if gensemantic:
        semantic_input_data = np.asarray(semantic_input_data)
    else:
        semantic_input_data = np.zeros((1,1))
    encoder_input_data = np.asarray(encoder_input_data)
    decoder_target_data = np.asarray(decoder_target_data)
    decoder_input_his = np.asarray(decoder_input_his)
    
    print(semantic_input_data.shape)
    
    return encoder_input_data, decoder_target_data, decoder_input_his, neighbor_values, neighbor_weight,semantic_input_data

def splitdata(data, testlen = test_len, shuffle = True):
    #split data into train and test set
    #data is a list of data(encoder,decoder,etc..), and each should has the same length
    #expect each data in datalist should be in the shape of [nodes, samples, obs_len, ..]
    #by default, we will use 7 days of the data for testing
    encoder_input_data, decoder_target_data, decoder_input_his,encoder_input_aux, neighbor_values, neighbor_weights, semantic_input_data = data
    train_encoder_input, train_decoder_target, train_decoder_input_his = encoder_input_data[:,:-testlen,], decoder_target_data[:,:-testlen,], decoder_input_his[:,:-testlen,]
    test_encoder_input, test_decoder_target, test_decoder_input_his = encoder_input_data[:,-testlen:,], decoder_target_data[:,-testlen:,], decoder_input_his[:,-testlen:,]
    train_encoder_input_aux,test_encoder_input_aux = encoder_input_aux[:,:-testlen,],encoder_input_aux[:,-testlen:,]
    train_neighbor_values,test_neighbor_values = neighbor_values[:,:-testlen,],neighbor_values[:,-testlen:,]
    train_neighbor_weights,test_neighbor_weights = neighbor_weights[:,:-testlen,],neighbor_weights[:,-testlen:,]
    train_semantic_input_data, test_semantic_input_data = semantic_input_data[:,:-testlen,],semantic_input_data[:,-testlen:,]
    
    train_decoder_input = np.zeros(train_decoder_target.shape)
    train_decoder_input[:,:,1:,] = train_decoder_target[:,:,:-1]
    train_decoder_input[:,:,0,] = train_encoder_input[:,:,-1,0]
    
    if (len(train_decoder_input.shape) == 3):
        train_decoder_input = np.expand_dims(train_decoder_input,axis = -1)
    if (len(train_decoder_target.shape) == 3):
        train_decoder_target = np.expand_dims(train_decoder_target, axis = -1)
    if (len(test_decoder_target.shape) == 3):
        test_decoder_target = np.expand_dims(test_decoder_target, axis = -1)

    return train_encoder_input_aux,test_encoder_input_aux,train_encoder_input, train_decoder_target, train_decoder_input_his, test_encoder_input, test_decoder_target, test_decoder_input_his, train_decoder_input, \
            train_neighbor_values,test_neighbor_values,train_neighbor_weights,test_neighbor_weights, train_semantic_input_data, test_semantic_input_data
    


if __name__ == '__main__':
   
    f = h5.File(data_file_path,'r')
    data = f['data']
    s = data.shape
    data = np.reshape(data,(s[0],s[1],s[2]*s[3]))
    t1,t2,t3, nei1, nei2, sem = makedataset(data,history = False, genneighbors = genneighbor, gensemantic = gensemantic, senmanpath = '../data/processed/walking_embedding_60min/')
    np.save('encoder_input_data.npy',t1)
    np.save('decoder_target_data.npy',t2)
    np.save('decoder_input_his.npy',t3)
    
    if genneighbors:
        np.save('neighbor_values.npy', nei1)
        np.save('neighbor_weights.npy', nei2)
    
    if gensemantic:
        np.save('semantic_input_data.npy', sem)
    
    if genaux:
        f2 = h5.File('../data/processed/Nov_call_data_t10_s3030_4070.h5','r')
        f3 = h5.File('../data/processed/Nov_sms_data_t10_s3030_4070.h5','r')
        data2 = f2['data'].value
        data3 = f3['data'].value
    
        s2 = data2.shape
        data2 = np.reshape(data2,(s2[0],s2[1],s2[2]*s2[3]))
        s3 = data3.shape
        data3 = np.reshape(data2,(s3[0],s3[1],s3[2]*s3[3]))
        data23 = np.concatenate([data2,data3],axis = 1)
        aux, _ , _, _, _,_= makedataset(data23,history = False,genneighbors = False, gensemantic = False) 
        print('data_aux',data23.shape)
        print('aux',aux.shape)
        np.save('encoder_input_aux.npy',aux)
 

    t1 = np.load('encoder_input_data.npy','r')
    t2 = np.load('decoder_target_data.npy','r')
    t3 = np.load('decoder_input_his.npy','r')
    t4 = np.load('encoder_input_aux.npy','r')
    t5 = np.load('neighbor_values.npy')
    t6 = np.load('neighbor_weights.npy')
    t7 = np.load('semantic_input_data.npy')
    daux,daux2,d1,d2,d3,d4,d5,d6,d7,nv1,nv2,nw1,nw2,sem1,sem2 = splitdata([t1,t2,t3,t4,t5,t6,t7])
    #return train_encoder_input_aux,test_encoder_input_aux,train_encoder_input, train_decoder_target, train_decoder_input_his, test_encoder_input, test_decoder_target, test_decoder_input_his, train_decoder_input
    #train_neighbor_values,test_neighbor_values,train_neighbor_weights,test_neighbor_weights
    
    
    if genh5File:
        f = h5.File('train_test_set_en{}_de{}_Nov_neighbor_new.h5'.format(obs_len, pre_len),'w')
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
        f.create_dataset('train_neighbor_values', data = nv1)
        f.create_dataset('test_neighbor_values', data = nv2)
        f.create_dataset('train_neighbor_weights', data = nw1)
        f.create_dataset('test_neighbor_weights', data = nw2)
        f.create_dataset('train_semantic_input_data', data = sem1)
        f.create_dataset('test_semantic_input_data', data = sem2)
        f.close()
        



