import h5py as h5
import time
import math
import numpy as np
file_ = '../data/processed/demo_internet_data.h5'#'../../data/processed/Nov_internet_data_t10_s3030_4070.h5'
f = h5.File(file_,'r')
data = f['data'].value
data = np.vstack(data)
data_4070 = data[:1008*3,:,:]
print(data_4070.shape)
interval = 1008

def cal_period_average(data, period = 1008):
    shape = [period]+list(data.shape[1:])
    s = np.zeros(shape)
    total_period = data.shape[0]
    for i in range(0, total_period, period):
        s += data[i:i+period,]
    step = total_period // period
    s /= step
    return s
    
def LB_Keogh(s2,s1,r = 6):
    LB_sum = 0
    for ind, i in enumerate(s1):
        s = s2[(ind - r if ind - r >= 0 else 0):(ind + r)]
        lb = min(s)
        ub = max(s)
        if i > ub:
            LB_sum = LB_sum+(i - ub) ** 2
        elif i < lb:
            LB_sum = LB_sum+(i - lb) ** 2
    return math.sqrt(LB_sum)

def gen_dis_matrix(grids, gridNum = 900, method ='lb',w = 6):#method = dte || lb
    #grids is list of grid series features, i.e. the temporal series of grids
    print('generating matrix...')
    str_time = time.time()
    dis_mat = np.zeros((gridNum,gridNum))
    if method == 'lb':
        func = LB_Keogh
    else:
        func = DTWdistance
    for i in range(gridNum):
        print('grid'+str(i)+'....')
        #here we deal the the dis matrix as undirected for simplicity
        #for j in range(i+1,gridNum):
        for j in range(gridNum):
            dis = func(grids[i],grids[j],w)
            dis_mat[i][j] = dis
        print('{} seconds been spent..'.format(time.time()-str_time))
    print('done...')
    return dis_mat

def gen_weight_from_dtw(dtw_mat):
    min_max_scaler = preprocessing.MinMaxScaler()
    dtw_minmax = min_max_scaler.fit_transform(dtw_mat)
    wei_mat = np.exp(-dtw_minmax)
    return wei_mat

def build(dataX, meth = 'lb'):
    nb_slots,nb_row,nb_col = dataX.shape
    nb_grid = nb_row*nb_col
    d = np.reshape(dataX,(nb_slots,nb_grid))
    grids = d.T
    dis_mat = gen_dis_matrix(grids,nb_grid,meth)
    #wei_mat = gen_weight_from_dtw(dis_mat)
    return dis_mat #,wei_mat
    
data_4070_weekly = cal_period_average(data_4070)
dis = build(data_4070_weekly,'lb')
np.save('dis_matrix_lb_weekly_4070_directed.npy',dis)