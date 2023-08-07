from __future__ import print_function, division
import os
import numpy as np
import h5py
from imblearn.over_sampling import SMOTE
import math

#########################
# division datasets, save same class
#########################
def __divide_dataset():
    # save file
    path = 'Data/DRK.txt'
    if not os.path.isfile(path):
        raise Exception('data not found!')

    # save class
    data = {}

    # divide dataset
    with open(path, 'r') as reader:
        for line in reader.readlines():
            line = line.strip()
            line = line.replace(',',' ')
            line = line.split()
            k = line[-1]
            v = np.asarray([float(x) for x in line[:-1]])
            v = v[np.newaxis, :]
            if k in data.keys():
                data[k] = np.concatenate((data[k], v), axis = 0)
            else:
                data[k] = v

    # sample number sort
    data = dict(sorted(data.items(), key = lambda item: len(item[1])))

    # save dataset
    with h5py.File('Data/DRK.h5', 'w') as hf:
        cnt = 0
        for k, v in data.items():
            hf[str(cnt)] = v
            print(cnt, k, len(v))
            cnt += 1

    print("dataset divided finish")

####################################################################
# satistic dataset, number of samples in each class, the max and min
#####################################################################
def __statistic_dataset():
    # all number of samples
    amount = 9900
    # number of features
    feature = 2
    # number of class
    category = 2
    # the max and min
    val = [[float("inf"), 0.] for i in range(feature)]

    # get the max and min
    with h5py.File('Data/DRK.h5', 'r') as hf:
        for i in range(category):
            data = np.asarray(hf[str(i)])
            print(i, data.shape[0], "{:.4f}".format(data.shape[0] / amount))
            for i in range(feature):
                val[i][0] = min(val[i][0], np.min(data[:,i]))
                val[i][1] = max(val[i][1], np.max(data[:,i]))

    # print the min and max
    for i in range(feature):
        print("[{}--{}]".format(val[i][0], val[i][1]), end=' ')
    print('')

#########################################################################
# 5-fold divition train val
#########################################################################
def __divide_train_val_dataset(ratio = 0.5, fold = 0):
    """
    :param ratio:
    :param fold:
    :return:
    """
    # number of sample class
    category = 2

    # divide train and val
    with h5py.File('Data/DRK.h5','r') as reader:
        # save val dataset
        with h5py.File('Data/DRT_val.h5', 'w') as writer:
            for i in range(category):
                # read data
                data = np.asarray(reader[str(i)])
                # length of data
                len = data.shape[0]
                # obtain number of sample
                stride = int(len * ratio)
                # save val
                writer[str(i)] = data[stride * fold : stride * (fold +1 )]

        # save train
        with h5py.File('Data/DRK_train.h5', 'w') as writer:
            for i in range(category):
                # read data
                data = np.asarray(reader[str(i)])
                # length of data
                len = data.shape[0]
                # obtain number of sample
                stride = int(len * ratio)
                # save train
                writer[str(i)] = np.concatenate((data[: stride * fold], data[stride * (fold + 1) :])
                                                , axis = 0)
    print("divide dataset finish")

###########################################################################
# main
###########################################################################
if __name__ == '__main__':
    # divide dataset
    __divide_dataset()

    # divede train and val
    __divide_train_val_dataset(ratio = 0.5, fold = 0)

    pass