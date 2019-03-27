import numpy as np

neighbnum = 15

def invweight(dist, num = 1., const = 0.1):
    return num/(dist + const)

data = np.load('dis_matrix_lb_weekly_4070.npy')
neighborsid = np.zeros(shape=(900,neighbnum), dtype = int)

for i in range(900):
    neighbor = np.argsort(data[i,:])[1:neighbnum + 1]
    neighborsid[i,:] = neighbor
print(neighborsid)

for i in range(900):
    for j in range(900):
        if j not in neighborsid[i,:]:
            data[i,j] = 0
        else:
            data[i,j] = invweight(data[i,j])
            
for i in range(900):
    exp_sum = 0
    for j in neighborsid[i,:]:
        j = int(j)
        exp_sum += (data[i,j])
    for j in range(900):
        if data[i,j] != 0:
            data[i,j] = (data[i,j])/exp_sum

np.save('./neighbor_weights_matrix_new.npy', data)

