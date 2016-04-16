import csv
import math
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd


def lerp(value_1, value_2, alpha=0.5):  # Linear interpolation
    if value_1 <= value_2:
        return value_1 + (value_2-value_1)*alpha
    else:
        return value_2 + (value_1-value_2)*(1-alpha)


def cerp(value_1, value_2, alpha=0):  # Cos interpolation
    if value_1 <= value_2:
        return value_1 + (value_2-value_1)*math.cos(lerp(0, math.pi / 2, (1 - alpha)))
    else:
        return value_2 + (value_1-value_2)*math.cos(lerp(0, math.pi / 2, alpha))


def average(values):
    return sum(values)/len(values)


def compress_data(data, ratio=.1):

    last_5 = []
    compressed_data = []

    for key, value in np.ndenumerate(data):

        if (key[0] % ratio**-1) == 0 and key[0] != 0:
            compressed_data.append(average(last_5))
            last_5 = []

        last_5.append(value)

    return compressed_data


def clamp(value, min_value=0, max_value=1):

    if min_value <= value <= max_value:
        return value
    elif value < min_value:
        return min_value
    else:
        return max_value


def split_data_set(data_set, targets, test_size=.5):
    return train_test_split(data_set, targets, test_size=test_size)


def smooth_data(y, smooth_factor=10):
    box = np.ones(smooth_factor) / smooth_factor
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def accuracy(predicted_targets, real_targets):
    """
    :param predicted_targets: Numpy array with predicted targets.
    :param real_targets: Numpy array with real targets.
    :return: Float value between 0 - 1, representing accuracy (ex. 0.5 = half of the targets where correct)
    """
    predicted_targets = predicted_targets.tolist()
    real_targets = real_targets.tolist()
    correct_sum = 0

    for i in range(len(predicted_targets)):
        if predicted_targets[i] == real_targets[i]:
            correct_sum += 1
    return correct_sum/len(predicted_targets)

def csv_to_data_set(file_name):

    data_set = []
    targets = []
    with open(file_name) as csv_file:
        data_file = csv.reader(csv_file)
        # temp = next(data_file)
        for row in data_file:
            data_set.append(row[:-1])
            targets.append(row[-1])

    return (np.array(data_set).astype(float), np.array(targets))


def remove_doubles(in_list):
    return list(set(in_list))


def show_confusion_matrix(test_targets, predicted_targets, labels):
    
    print("-"*30, "Confusion matrix", "-"*30)

    print(("{: >20}"*(len(labels)+1)).format("", *labels))
    
    for key, value in enumerate(confusion_matrix(test_targets, predicted_targets, labels)):
        
        print(("{: >20}"*(len(value)+1)).format(labels[key], *value))
