import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks



def difference_mean(a):
    total = 0

    for i in range(len(a) - 1):
        total += a[i + 1] - a[i]

    return total / (len(a) - 1)

def index_correction(a, mean):
    # a -> original index list
    corrected_index = []
    counter = 0

    for i in range(len(a) - 1):
        # print(counter)
        if counter < (len(a) - 1):
            if a[counter + 1] - a[counter] < mean * 0.9:
                corrected_index.append(a[counter])
                counter += 1
                # print('index corrected!!!')
            else:
                corrected_index.append(a[counter])

        counter += 1

    return corrected_index


def total_index_gen(corrected_peak, corrected_valley):
    total_index = []
    total_index.extend(corrected_peak)
    total_index.extend(corrected_valley)
    total_index = sorted(total_index)

    return total_index


def find_characteristic(total_index, corrected_peak, corrected_valley):
    char_index = []
    peak_reg = 0
    peak_counter = 0
    valley_reg = 0
    valley_counter = 0

    for n in range(len(total_index)):
        if total_index[n] in corrected_peak:
            peak_reg = total_index[n]
            peak_counter += 1
            if peak_counter > 1:
                peak_counter = 0
        elif total_index[n] in corrected_valley:
            valley_reg = total_index[n]
            valley_counter += 1
            if valley_counter > 1:
                valley_counter = 1

        if peak_counter == 1 and valley_counter == 1:
            if peak_reg < valley_reg:       # means peaks is in front of valley
                peak_counter = 0
                # valley_counter = 0
            elif peak_reg > valley_reg:     # means normal situation
                char_index.append(int((peak_reg + valley_reg) / 2))

                peak_counter = 0
                valley_counter = 0

    return char_index


def find_shorter(a, b):
    if a > b:
        return b
    elif a < b:
        return a


def PTT_calc(ECG_peaks, char_points):
    PTT = []
    PTT_index = []
    ECG_index = []
    char_index = []
    ECG_counter = 0
    char_counter = 0
    interval = 10

    # length = find_shorter(len(ECG_peaks), len(char_points))

    while(True):
        if abs(ECG_peaks[ECG_counter] - char_points[char_counter]) <= interval or ECG_peaks[ECG_counter] >= char_points[char_counter]:
                char_counter += 1
        elif abs(ECG_peaks[ECG_counter] - char_points[char_counter]) > interval:
            PTT_value = abs(ECG_peaks[ECG_counter] - char_points[char_counter])
            PTT.append(PTT_value)
            PTT_index.append(ECG_peaks[ECG_counter])

            ECG_index.append(ECG_peaks[ECG_counter])
            char_index.append(char_points[char_counter])

            ECG_counter += 1
            # char_counter += 1

        if char_counter > len(char_points) - 1 or ECG_counter > len(ECG_peaks) - 1:
            break

    return PTT, PTT_index, ECG_index, char_index