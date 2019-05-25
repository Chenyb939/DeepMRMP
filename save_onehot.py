"""
# -*- coding: utf-8 -*-
# @Time    : 2018/6/1 11:42
# @Author  : Chenyb
# @Email   : 1041429151@qq.com
# @File    : save_onehot.py
"""

import pickle
import numpy as np
import data_processing as dp


def onehotkey(seq, tag):
    tag = np.array(tag)
    for num in range(len(seq)):
        seq[num] = seq[num].strip('\n')

    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["G"] = 2
    letterDict["U"] = 3
    letterDict["T"] = 3
    CategoryLen = 4
    probMatr = np.zeros((len(seq), 1, 41, CategoryLen))
    # probMatr = np.zeros((len(seq), 1, len(seq[0]), CategoryLen))
    sampleNo = 0

    for sequence in seq:
        RNANo = 0
        sequence = sequence.center(51, 'N')
        for RNA in sequence:
            try:
                index = letterDict[RNA]
                probMatr[sampleNo][0][RNANo][index] = 1
                RNANo += 1
            except:
                RNANo += 1
        sampleNo += 1
    return probMatr, tag


def save_onehot(folder_path, savepath, copy):
    copies = copy + 1
    for i in range(1, copies):
        times = i
        train = dp.fa_to_df(folder_path + '/train%d.fa' % times)
        val = dp.fa_to_df(folder_path + '/val%d.fa' % times)
        test = dp.fa_to_df(folder_path + '/test%d.fa' % times)

        seq, label = dp.getseq(train)
        probMatr, tag = onehotkey(seq, label)
        trainX = probMatr
        trainY = tag

        fileX = open(savepath + '/trainX%d.pickle' % times, 'wb')
        fileY = open(savepath + '/trainY%d.pickle' % times, 'wb')
        pickle.dump(trainX, fileX, protocol=4)
        pickle.dump(trainY, fileY, protocol=4)
        fileX.close()
        fileY.close()

        seq, label = dp.getseq(val)
        probMatr, tag = onehotkey(seq, label)
        trainX = probMatr
        trainY = tag

        fileX = open(savepath + '/valX%d.pickle' % times, 'wb')
        fileY = open(savepath + '/valY%d.pickle' % times, 'wb')
        pickle.dump(trainX, fileX, protocol=4)
        pickle.dump(trainY, fileY, protocol=4)
        fileX.close()
        fileY.close()

        seq, label = dp.getseq(test)
        probMatr, tag = onehotkey(seq, label)
        testX = probMatr
        testY = tag

        fileX = open(savepath + '/testX%d.pickle' % times, 'wb')
        fileY = open(savepath + '/testY%d.pickle' % times, 'wb')
        pickle.dump(testX, fileX, protocol=4)
        pickle.dump(testY, fileY, protocol=4)
        fileX.close()
        fileY.close()

        print('one hot time %d has been saved' % times)


def split_data(pos_path, neg_path, save_path, pos_rate, neg_rate, copy):
    copies = copy + 1

    pos_df = dp.fa_to_df(pos_path)
    neg_df = dp.fa_to_df(neg_path)
    pos_df = pos_df.sample(frac=1)
    neg_df = neg_df.sample(frac=1)
    pos_df = pos_df[0: int(len(pos_df) * pos_rate)]
    neg_df = neg_df[0: int(len(pos_df) * neg_rate)]

    for i in range(6, copies):
        times = i
        pos_train, pos_val, pos_test = dp.per_split(pos_df, 0.6, 0.2)
        neg_train, neg_val, neg_test = dp.per_split(neg_df, 0.6, 0.2)
        train = pos_train.append(neg_train)
        train = train.sample(frac=1)
        val = pos_val.append(neg_val)
        val = val.sample(frac=1)
        test = pos_test.append(neg_test)
        test = test.sample(frac=1)
        dp.df_to_fa(train, save_path + '/train%d.fa' % times)
        dp.df_to_fa(val, save_path + '/val%d.fa' % times)
        dp.df_to_fa(test, save_path + '/test%d.fa' % times)
        print('split data time %d has been saved' % times)


def main(type):

    save_onehot('./data_train/%s' % type + '/fa', './data_train/%s' % type + '/onehot', 10)


if __name__ == '__main__':
    types = ['m1A', 'm5C', 'Pseu']
    for type in types:
        main(type)
