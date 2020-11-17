# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:27:56 2020

@author: Silence
"""


import h5py
import numpy as np
import matplotlib as plt



ratio_tr = 0.7
ratio_cv = 0.2
ratio_te = 0.1

path_file = 'L:/work/Radio_Modulation_Recognition/data/data.hdf5'


data_dict = {}
        
with h5py.File(path_file, 'r') as hf:
    data_all = hf['X'][:]
    label_all = hf['Y'][:]
    snr_all = hf['Z'][:]
    
n_sample = len(data_all)
n_class = np.size(label_all, 1)
n_snr = len(np.unique(snr_all))

n_sample_each_class = n_sample//n_class
index_tmp = np.zeros((n_class,n_sample_each_class),dtype = int)
for i in range(n_class):
    index_tmp[i,:] = i*n_sample_each_class + np.random.permutation(n_sample_each_class)
index_tr = index_tmp[:,0:round(ratio_tr*n_sample_each_class)]
index_cv = index_tmp[:,round(ratio_tr*n_sample_each_class):round((ratio_tr+ratio_cv)*n_sample_each_class)]
index_te = index_tmp[:,round((ratio_tr+ratio_cv)*n_sample_each_class):]
index_tr = index_tr.flatten()
index_cv = index_cv.flatten()
index_te = index_te.flatten()

data_tr = data_all[index_tr]
data_cv = data_all[index_cv]
data_te = data_all[index_te]
del data_all

label_tr = label_all[index_tr]
label_cv = label_all[index_cv]
label_te = label_all[index_te]
del label_all

snr_tr = snr_all[index_tr]
snr_cv = snr_all[index_cv]
snr_te = snr_all[index_te]
del snr_all









#%%
# with open("/home/hzy/keras/RML2016.10a_dict.dat",'rb') as xd1: #这段执行对原始数据进行切片的任务，可在spyder下运行，查看变量
#     Xd = pickle.load(xd1,encoding='latin1')
# snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
# X = []  
# lbl = []
# for mod in mods:
#     for snr in snrs:
#         X.append(Xd[(mod,snr)])
#         for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
# X = np.vstack(X)