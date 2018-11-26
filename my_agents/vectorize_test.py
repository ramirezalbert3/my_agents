import numpy as np
import random
import time

x = 2   # n_actions
y = 32  # batch_size
z = 21  # n_atoms

m1 = np.zeros((x,y,z))
m2 = np.zeros((x,y,z))
m3 = np.zeros((x,y,z))

time1 = []
time2 = []
time3 = []

vals = np.ones((y,z))

for i in range(400):
    act = np.array([random.randint(0,x-1) for _ in range(y)])
    m = np.array([random.randint(0,z-1) for _ in range(y*z)]).reshape((y,z))

    ii = np.arange(y)
    jj = np.arange(z)

    start = time.time()
    for i in range(y):
        for j in range(z):
            m1[act[i], i, m[i,j]] += vals[i,j]
    time1.append(time.time()-start)
    
    start = time.time()
    for i in range(y):
        m2[act[i], i, m[i,jj]] += vals[i]
    time2.append(time.time()-start)
    
    start = time.time()
    t = np.zeros(m3[act, ii].shape)
    np.put_along_axis(t, m, vals, axis=1)
    m3[act, ii] += t
    time3.append(time.time()-start)
    
    if not np.array_equal(m1, m2):
        print(m1, '\n\nm1 not equal to m2\n\n', m2)
        raise AssertionError()
    
    if not np.array_equal(m2, m3):
        print(m2, '\n\nm2 not equal to m3\n\n', m3)
        raise AssertionError()
    
    # print('m1\n',m1,'\n')
    # print('m2\n',m2,'\n')
    # print('m3\n',m3,'\n')

mean1 = np.mean(time1)
mean2 = np.mean(time2)
mean3 = np.mean(time3)
tmax = max(mean1, mean2, mean3)
print('vectorization successful!')
print('average time1: {:.3}s - {:.4}%'.format(mean1, 100*mean1/tmax))
print('average time2: {:.3}s - {:.4}%'.format(mean2, 100*mean2/tmax))
print('average time3: {:.3}s - {:.4}%'.format(mean3, 100*mean3/tmax))
