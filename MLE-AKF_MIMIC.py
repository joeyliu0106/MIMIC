import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from MLE_AKF_functions import *
import pandas as pd
from statistics import mean
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

index = 0   # file index(0~5, 'all')

######   data reading   ######
root = os.getcwd()
file_path = os.path.join(root, 'data', 'MLE-AKF data')
os.chdir(file_path)


# continuous
SBP = np.load('SBP.npy')
print('SBP shape: ', SBP.shape)
HR = np.load('HR.npy')
print('HR shape: ', HR.shape)
PTT = np.load('Filtered_PTT.npy')
print('PTT shape: ', PTT.shape)

X = np.zeros((3, len(SBP)), dtype=float)
X[0] = [1 for i in range(len(SBP))]
X[1] = [PTT[i] for i in range(len(SBP))]
X[2] = [HR[i] for i in range(len(SBP))]
print(X[:, 0].reshape(3, 1))


######   parameters   ######
akf_counter = 0
mle_end_flag = 0
akf_exe_flag = 0
calibration_period = 60    # calibration interval  # original: 70
registration_num = 25       # calibration number for MLE

A = np.zeros((3, 3), dtype=float)
a = np.zeros((3, 1), dtype=float)
sig_c = 0

result_list = []
result_index_list = []
error = []

######   MLE-AKF   ######
for n in range(len(SBP)):
    print('sample number: ', n + 1)
    input = X[:, n].reshape((3, 1))

    # coefficient calculation
    if n < registration_num:
        # MLE
        coeff, A, a = MLE(input, SBP[n], A, a)

        if n == registration_num - 1:
            mle_end_flag = 1
            print('MLE ended')

    # decide if AKF should work or not
    if mle_end_flag == 1 or ((n + 1) - registration_num) % calibration_period == 0:
        akf_exe_flag = 1
        mle_end_flag = 0
        print('AKF activated')

    # executing AKF
    if akf_exe_flag == 1:
        # previous coefficient remembering
        coeff_before = np.copy(coeff)

        if akf_counter == 0:
            sig_r = np.dot(X[:, 0].reshape((3, 1)), X[:, 0].reshape((3, 1)).T)
        sig_r = pinv(pinv(sig_r + sig_c) + np.dot(input, input.T))

        coeff = AKF(coeff, input, sig_r, SBP[n])
        # sig_c update
        sig_c = coeff_cov(coeff_before, coeff)
        akf_counter = akf_counter + 1
        akf_exe_flag = 0

    if akf_counter > 0:
        result = np.dot(coeff.T, input)
        result_list.append(result)
        result_index_list.append(n)

        print('golden: ', SBP[n])
        print('estimated: ', result)

result_list = np.array(result_list)
result_list = np.reshape(result_list, len(result_index_list))

######   averaging   ######
avg_num = 5
avg_SBP_list = []
avg_result_list = []
avg_result_index = []

for i in range(len(result_list)):
    if i % avg_num == 0:
        avg_result = mean(result_list[i:(i + avg_num - 1)])
        avg_result_list.append(avg_result)
        avg_result_index.append(result_index_list[i])

        avg_SBP = mean(SBP[i:(i + avg_num - 1)])
        avg_SBP_list.append(avg_SBP)


######   visualization   ######
x = list(range(0, len(result_list)))
x_SBP = list(range(0, len(SBP)))


plt.figure()
plt.xlabel('Samples')
plt.ylabel('SBP value')
plt.title('Estimating Result')
# plt.plot(x, result_list, label='whole estimated')
# plt.plot(x_SBP, SBP, label='golden')
plt.plot(avg_result_index, avg_result_list, label='avg estimated')
plt.plot(avg_result_index, avg_result_list, 'ko')
plt.plot(avg_result_index, avg_SBP_list, label='avg golden')
plt.plot(avg_result_index, avg_SBP_list, 'ko')
plt.legend()


plt.figure()
plt.plot(PTT, 'x', label='PTT')
plt.title('Pulse Transit Time')
plt.legend()
# ax1.plot(ECG_peaks, ECG[ECG_peaks], "x")

plt.figure()
plt.plot(SBP, 'x', label='SBP')
plt.title('Systolic Blood Pressure')
plt.legend()

plt.figure()
plt.plot(HR, 'x', label='HR')
plt.title('Heart Rate')
plt.legend()

plt.show()


######   criteria print out   ######
print('#####################################################')
mse = mean_squared_error(avg_SBP_list, avg_result_list)
rmse = sqrt(mean_squared_error(avg_SBP_list, avg_result_list))
print('MSE: ', mse)
print('RMSE: ', rmse)
print('correlation matrix:\n', np.corrcoef(avg_SBP_list, avg_result_list))