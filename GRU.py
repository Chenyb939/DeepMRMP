"""
# -*- coding: utf-8 -*-
# @Time    : 2018/6/1 11:42
# @Author  : Chenyb
# @Email   : 1041429151@qq.com
# @File    : GRU.py
"""
# -*- coding: GBK -*-

import time
import pickle
import matplotlib
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler, History, TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1_l2
import keras.metrics
# from PyQt4 import QtCore
import matplotlib.pyplot as plt
from keras.optimizers import Nadam, Adam, SGD
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
import data_processing
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
import configparser

# import xlwt
# from xlrd import open_workbook
# from xlutils.copy import copy

config = configparser.ConfigParser()
config.read('config.txt', encoding='utf-8')

matplotlib.use('Agg')
plt.switch_backend('agg')

gpu_id = '5, 7'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)


def model(input):
    #  ######### First Network ##########
    x = GRU(units=64, return_sequences=True)(input)
    x = Dropout(0.2)(x)

    x = GRU(units=64, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    output = core.Flatten()(x)
    output = BatchNormalization()(output)

    out = Dense(64, activation="relu", kernel_initializer='glorot_normal')(output)
    out = Dense(2, activation="softmax", kernel_initializer='glorot_normal')(out)

    #  ########## Set Cnn ##########
    cnn = Model(inputs=input, outputs=out)
    cnn.summary()
    adam = keras.optimizers.Adam()
    cnn.compile(loss='binary_crossentropy', optimizer=adam, metrics=[keras.metrics.binary_accuracy])  # Nadam
    return cnn


def fit_model(trainX, trainY, valX, valY, cnn, i, type):
    checkpointer = ModelCheckpoint(filepath='./fig1/%s' % type + '/without/model/' + str(type) + '_%d.h5' % i, verbose=1,
                                   save_best_only=True, monitor='val_loss', mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    fitHistory = cnn.fit(trainX, trainY, batch_size=1024, epochs=1000, verbose=2, validation_data=(valX, valY),
                         callbacks=[checkpointer, earlystopping])
    return fitHistory


def fine_tuning(m1AtrainX, m1AtrainY, m1AvalX, m1AvalY, i, type, modelname):
    model = models.load_model(modelname)
    for layer in model.layers:
        layer.trainable = True

    adam = Adam(lr=0.00005)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[keras.metrics.binary_accuracy])
    checkpointer = ModelCheckpoint(filepath='./fig1/%s' % type + '/without/model/' + str(type) + '_%d.h5' % i, verbose=1,
                                   save_best_only=True, monitor='val_loss', mode='min')

    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=200)
    fitHistory = model.fit(m1AtrainX, m1AtrainY, batch_size=128, epochs=500000, verbose=2,
                           validation_data=(m1AvalX, m1AvalY), callbacks=[checkpointer, earlystopping])
    # class weight 选项balance, auto
    return fitHistory


def evaluate_model(modelname, testX, testY, i, type):
    cnn = models.load_model(modelname)
    # cnn = models.load_model('%d-merge.h5' % i, {'isru': isru, 'pearson_r': pearson_r})
    #  ############### test ##########################
    pre_score = cnn.evaluate(testX, testY, batch_size=2048, verbose=0)

    # fileX = open('./fig1/%s' % type + '/without/pre_score%d.pickle' % i, 'wb')
    # pickle.dump(pre_score, fileX, protocol=4)
    # fileX.close()

    # 最后做对比图写出来
    #  ######### Print Precision and Recall ##########
    pred_proba = cnn.predict(testX, batch_size=2048)

    fileX = open('./fig1/%s' % type + '/without/pred_proba%d.pickle' % i, 'wb')
    pickle.dump(pred_proba, fileX, protocol=4)
    fileX.close()

    pred_score = pred_proba[:, 1]
    true_class = testY[:, 1]

    precision, recall, _ = precision_recall_curve(true_class, pred_score)
    average_precision = average_precision_score(true_class, pred_score)

    fpr, tpr, thresholds = roc_curve(true_class, pred_score)
    roc_auc = auc(fpr, tpr)

    for index in range(len(pred_score)):
        if pred_score[index] > config.getfloat('others', 'threshold'):
            pred_score[index] = 1
        else:
            pred_score[index] = 0

    mcc = matthews_corrcoef(true_class, pred_score)

    plt.figure()
    plt.step(recall, precision, color='navy', where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('./fig1/%s' % type + '/without/curve/' + str(type) + 'Precision-Recall%d.png' % i)

    #  ################# Print ROC####################

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Inception ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('./fig1/%s' % type + '/without/curve/' + str(type) + 'ROC %d.png' % i)
    SN, SP = performance(true_class, pred_score)
    pre = precision_score(y_true=true_class, y_pred=pred_score)
    rec = recall_score(y_true=true_class, y_pred=pred_score)
    f1 = f1_score(y_true=true_class, y_pred=pred_score)

    # Sn和recall是同一个值
    return pre_score, pre, rec, SN, SP, f1, mcc, roc_auc


def print_loss(fitHistory, i, type):
    #  ######### Print Loss Map ##########
    plt.figure()
    plt.plot(fitHistory.history['loss'][:-20])
    plt.plot(fitHistory.history['val_loss'][:-20])
    # plt.title('size:%d' % size)
    plt.title('LOSS:times %d' % i)
    plt.ylim([0, 1.0])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./fig1/%s' % type + '/without/curve/' + str(type) + 'loss%d.png' % i)

    #  ############### final ################
    loss1 = fitHistory.history['loss'][-21:-20]
    acc1 = fitHistory.history['binary_accuracy'][-21:-20]
    loss2 = fitHistory.history['val_loss'][-21:-20]
    acc2 = fitHistory.history['val_binary_accuracy'][-21:-20]

    return loss1, acc1, loss2, acc2


def print_fine_loss(fitHistory, i, type):
    #  ######### Print Loss Map ##########
    plt.figure()
    plt.plot(fitHistory.history['loss'][:-200])
    plt.plot(fitHistory.history['val_loss'][:-200])
    # plt.title('size:%d' % size)
    plt.title('LOSS:times %d' % i)
    plt.ylim([0, 1.0])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./fig1/%s' % type + '/without/curve/' + str(type) + 'loss%d.png' % i)

    #  ############### final ################
    loss1 = fitHistory.history['loss'][-201:-200]
    acc1 = fitHistory.history['binary_accuracy'][-201:-200]
    loss2 = fitHistory.history['val_loss'][-201:-200]
    acc2 = fitHistory.history['val_binary_accuracy'][-201:-200]

    return loss1, acc1, loss2, acc2


def performance(labelArr, predictArr):
    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    SN = TP / (TP + FN)
    SP = TN / (FP + TN)
    return SN, SP


def to_int(input):
    input = input.astype(np.float32)
    if input == 1.0:
        input = int(1)
    elif input == 0.0:
        input = int(0)
    return input


def main(type):
    if os.path.exists('./fig1/%s' % type + '/without/results.txt'):
        print("The result will be written in the 'results.txt' file")
    else:
        with open('./fig1/%s' % type + '/without/results.txt', 'w') as file:
            file.write('date' + '\t' + 'type' + '\t' + 'time' + '\t' + 'train loss' + '\t' + 'train acc' + '\t' +
                       'val loss' + '\t' + 'val acc' + '\t' + 'test loss' + '\t' + 'test acc' + '\t' + 'precision' +
                       '\t' + 'recall' + '\t' + 'Sn' + '\t' + 'Sp' + '\t' + 'F1' + '\t' + 'mcc' + '\t' + 'auc' + '\n')

    for i in range(8, 9):
        trainX, trainY, input, _ = data_processing.load_data('./data_train/%s' % type + '/onehot/trainX%d.pickle' % i)
        valX, valY, _, _ = data_processing.load_data('./data_train/%s' % type + '/onehot/valX%d.pickle' % i)
        testX, testY, _, _ = data_processing.load_data('./data_train/%s' % type + '/onehot/testX%d.pickle' % i)
        # testX, testY, _, _ = data_processing.load_data('./m1Aonehot/trainX%d.pickle' % i)

        # cnn = model(input)
        # history = fit_model(trainX, trainY, valX, valY, cnn, i, type)
        # # history = fine_tuning(trainX, trainY, valX, valY, i, type, './m6A/GRU/1_m6A.h5')
        #
        # loss1, acc1, loss2, acc2 = print_loss(history, i, type)
        # print('train loss:', loss1,
        #       'train acc:', acc1,
        #       'val loss:', loss2,
        #       'val acc:', acc2)

        loss1, acc1, loss2, acc2 = '0', '0', '0', '0'

        pre_score, pre, rec, SN, SP, f1, mcc, roc_auc = evaluate_model('./fig1/%s' % type + '/without/model/' + str(type) + '_%d.h5' % i,
                                                                       testX, testY, i, type)

        # loss1, acc1, loss2, acc2 = 0, 0, 0, 0,
        # pre_score, pre, rec, SN, SP, f1, mcc, roc_auc = evaluate_model('./model/m1A_3.h5', testX,
        #                                                                testY, i, type)

        print('test loss:', pre_score[0],
              'test acc:', pre_score[1])

        with open('./fig1/%s' % type + '/without/results.txt', 'a') as file:
            file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\t' + str(type) + '\t' + str(i) + '\t' +
                       str(loss1) + '\t' + str(acc1) + '\t' + str(loss2) + '\t' + str(acc2) + '\t' +
                       str(pre_score[0]) + '\t' + str(pre_score[1]) + '\t' + str(pre) + '\t' + str(rec) + '\t' +
                       str(SN) + '\t' + str(SP) + '\t' + str(f1) + '\t' + str(mcc) + '\t' + str(roc_auc) + '\n')


if __name__ == '__main__':

    # types = ['m1A', 'm5C', 'Pseu']
    # for type in types:
    #     main(type)
    main('Pseu')
