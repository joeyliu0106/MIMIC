import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from statistics import *
from functions import *

#file location
file_name = 'Part_1'
data_path = os.path.join('data', file_name)
# data list & index
dir_list = os.listdir(data_path)
data_index = 1          # 0~2999
print(dir_list)

os.chdir(data_path)

# load MIMIC II data
data = np.load(dir_list[data_index])
PPG = data[0]
ABP = data[1]
ECG = data[2]

# ECG peaks
ECG_peaks, _ = find_peaks(ECG, height=1)
print(len(ECG_peaks))
# PPG peaks
PPG_peaks, _ = find_peaks(PPG, height=2)
print(len(PPG_peaks))
# PPG valley
PPG_valleys, _ = find_peaks(PPG*-1, height=-2)
print(len(PPG_valleys))
# ABP peak
ABP_peaks, _ = find_peaks(ABP, height=110)
print(len(ABP_peaks))

# ABP peak correction
corrected_ABP_peak = []
counter = 0

for n in range(len(ABP_peaks)):
    if counter < len(ABP_peaks) - 1:
        if abs(ABP_peaks[counter] - ABP_peaks[counter + 1]) < 20:
            corrected_ABP_peak.append(ABP_peaks[counter])
            counter += 2
        else:
            corrected_ABP_peak.append(ABP_peaks[counter])
            counter += 1
print('length of corrected ABP peak: ', len(corrected_ABP_peak))



# peak_mean = difference_mean(PPG_peaks)
# print('peak difference mean value: ', peak_mean)
# corrected_peak = index_correction(PPG_peaks, peak_mean)
# print('length of corrected peak: ', len(corrected_peak))
#
# valley_mean = difference_mean(PPG_valleys)
# print('valley difference mean value: ', valley_mean)
# corrected_valley = index_correction(PPG_valleys, valley_mean)
# print('length of corrected valley: ', len(corrected_valley))
#
# # new: with mean thresholding(index correction)
# total_index = total_index_gen(corrected_peak, corrected_valley)
# print('length of total index: ', len(total_index))
# char_list = find_characteristic(total_index, corrected_peak, corrected_valley)
# print('length of characteristic point index: ', len(char_list))


# old: without mean thresholding(index correction)
total_index_ori = total_index_gen(PPG_peaks, PPG_valleys)
print('length of total index: ', len(total_index_ori))
char_list_ori = find_characteristic(total_index_ori, PPG_peaks, PPG_valleys)
print('length of characteristic point index: ', len(char_list_ori))



# PTT calculation
print('length of ECG peaks: ', len(ECG_peaks))
print('length of characteristic points: ', len(char_list_ori))
PTT, PTT_index, ECG_index, char_index = PTT_calc(ECG_peaks, char_list_ori)
print('length of PTT: ', len(PTT))


corrected_PTT = []
corrected_PTT_index = []
time_stamp = []

std = stdev(PTT)
print(std)

for n in range(len(PTT)):
    if PTT[n] <= mean(PTT) * 1.1 and PTT[n] >= mean(PTT) * 0.9:
    # if PTT[n] <= mean(PTT) + std and PTT[n] >= mean(PTT) - std:
        corrected_PTT.append(PTT[n] * 0.008)
        corrected_PTT_index.append(PTT_index[n])
        time_stamp.append(PTT_index[n] * 0.008)


print('length of corrected PTT: ', len(corrected_PTT))
print(corrected_PTT)


######   SBP finding   ######
counter = 0
SBP = []

for n in range(len(corrected_PTT_index)):
    if counter < len(corrected_ABP_peak):
        while(ECG_peaks[n] > corrected_ABP_peak[counter]):
            counter += 1

    SBP.append(ABP[corrected_ABP_peak[counter]])

print('length of SBP: ', len(SBP))
print(SBP)

######   HR calculation   ######
HR = []

for n in range(len(corrected_PTT_index)):
    # index = ECG_peaks.index(corrected_PTT_index[n])
    index = np.where(ECG_peaks == corrected_PTT_index[n])[0][0]
    print(index)
    heart_rate = round(60 / ((ECG_peaks[index + 1] - ECG_peaks[index]) * 0.008))
    print(heart_rate)
    print(ECG_peaks[index + 1] - ECG[index])

    HR.append(heart_rate)

print(len(HR))
print(HR)


######   data saving   ######
saving_data = np.zeros((4, len(SBP)), float)

for n in range(len(SBP)):
    saving_data[0, n] = time_stamp[n]
    saving_data[1, n] = corrected_PTT[n]
    saving_data[2, n] = HR[n]
    saving_data[3, n] = SBP[n]

np.save('data.npy', saving_data)



# visualization
plt.figure(0)
plt.plot(ECG)
plt.plot(PPG)
plt.plot(char_list_ori, PPG[char_list_ori], "x")
plt.plot(ECG_peaks, ECG[ECG_peaks], "x")

plt.figure(1)
plt.plot(PPG)
plt.plot(PPG_peaks, PPG[PPG_peaks], "x")                # peak
plt.plot(PPG_valleys, PPG[PPG_valleys], "x")            # valley
# plt.plot(char_list_ori, PPG[char_list_ori], "x")                # char_list

# plt.figure(2)
# plt.plot(PPG)
# plt.plot(corrected_peak, PPG[corrected_peak], "x")      # corrected peak
# plt.plot(corrected_valley, PPG[corrected_valley], "x")  # corrected valley

plt.figure(3)
plt.plot(PPG)
plt.plot(PPG_valleys, PPG[PPG_valleys], "x")

# plt.figure(4)
# plt.plot(PPG)
# plt.plot(char_list, PPG[char_list], "x")

plt.figure(5)
plt.plot(PTT_index, PTT)
plt.plot(range(len(ABP)), ABP)


fig, ax1 = plt.subplots()
ax1.plot(ECG)
ax1.plot(PPG)
ax1.plot(char_list_ori, PPG[char_list_ori], "x")
ax1.plot(ECG_peaks, ECG[ECG_peaks], "x")

ax2 = ax1.twinx()
ax2.plot(corrected_PTT_index, corrected_PTT, c='black')
# ax2.plot(corrected_PTT_index, corrected_PTT, "x", c='black')
fig.tight_layout()


fig, ax1 = plt.subplots()
ax1.plot(ECG, c='black')
ax1.plot(ECG_peaks, ECG[ECG_peaks], "x")

ax2 = ax1.twinx()
ax2.plot(range(len(ABP)), ABP)
ax2.plot(corrected_ABP_peak, ABP[corrected_ABP_peak], 'x')
fig.tight_layout()


plt.figure()
plt.plot(corrected_PTT_index, corrected_PTT, label='PTT')
plt.legend()

plt.show()