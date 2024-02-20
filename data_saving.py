import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import h5py
import mat73
import os

file_name = 'Part_4'
saving_path = os.path.join('data', file_name)

if not os.path.isdir(os.path.join('data')):
    os.mkdir(os.path.join('data'))
if not os.path.isdir(saving_path):
    os.mkdir(saving_path)

mat = mat73.loadmat(f'{file_name}.mat')
data = mat[file_name]
# data = np.array(data)

os.chdir(saving_path)

for i in range(len(data)):
    np.save(f'{file_name}_{i}.npy', data[i])