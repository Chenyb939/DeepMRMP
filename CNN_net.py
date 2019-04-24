"""
# -*- coding: utf-8 -*-
# @Time    : 2018/6/1 11:42
# @Author  : Chenyb
# @Email   : 1041429151@qq.com
# @File    : CNN.py
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
from keras.layers import Dense, Dropout, Activation, Flatten, Input, GRU
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
gpu_id = '4,5,6,7'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

plt.switch_backend('agg')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)


def model(input, aa, numm):
    #  ######### First Network ##########
    # x = conv.Conv1D(config.getint('first layer', 'filters'), config.getint('first layer', 'kernel_size'),
    x = conv.Conv1D(config.getint('first layer', 'filters'), aa,
                    activation=config.get('first layer', 'activation'),
                    kernel_initializer=config.get('first layer', 'kernel_initializer'),
                    kernel_regularizer=l2(config.getfloat('first layer', 'kernel_regularizer')),
                    padding=config.get('first layer', 'padding'), name='Conv1')(input)
    x = Dropout(config.getfloat('first layer', 'dropout'), name='drop1')(x)
    x = BatchNormalization()(x)

    x = conv.Conv1D(config.getint('second layer', 'filters'), numm,
                    activation=config.get('second layer', 'activation'),
                    kernel_initializer=config.get('second layer', 'kernel_initializer'),
                    kernel_regularizer=l2(config.getfloat('second layer', 'kernel_regularizer')),
                    padding=config.get('second layer', 'padding'), name='Conv2')(x)
    x = Dropout(config.getfloat('second layer', 'dropout'), name='drop2')(x)
    # x = BatchNormalization()(x)

    output = core.Flatten()(x)
    output = BatchNormalization()(output)
    # output = Dropout(config.getint('flatten layer', 'dropout'), name='dropo3')(output)

    # output = Dense(config.getint('first dense layer', 'units'),
    #                kernel_initializer=config.get('first dense layer', 'kernel_initializer'),
    #                activation=config.get('first dense layer', 'activation'), name='Denseo1')(output)
    # output = Dropout(config.getfloat('first dense layer', 'dropout'), name='dropo4')(output)
    # output = BatchNormalization()(output)
    out = Dense(2, activation="softmax", kernel_initializer='glorot_normal', name='Denseo2')(output)

    #  ########## Set Cnn ##########
    cnn = Model(inputs=input, outputs=out)
    cnn.summary()
    adam = Adam(lr=0.0005)
    cnn.compile(loss='binary_crossentropy', optimizer=adam, metrics=[keras.metrics.binary_accuracy])  # Nadam
    return cnn


def fit_model(trainX, trainY, valX, valY, cnn, i, type):
    checkpointer = ModelCheckpoint(filepath='./cnn/model/' + str(type) + '%d_.h5' % i, verbose=1,
                                   save_best_only=True, monitor='val_loss', mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    fitHistory = cnn.fit(trainX, trainY, batch_size=512, epochs=5000, verbose=2, validation_data=(valX, valY),
                         callbacks=[checkpointer, earlystopping])
    return fitHistory


def fine_tuning(m1AtrainX, m1AtrainY, m1AvalX, m1AvalY, i, type, modelname):
    model = models.load_model(modelname)
    for layer in model.layers:
        layer.trainable = True

    adam = Adam(lr=0.00005)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[keras.metrics.binary_accuracy])
    checkpointer = ModelCheckpoint(filepath='./cnn/model/%d_' + str(type) + '.h5' % i, verbose=1,
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

    fileX = open('./cnn/pre_score/pre_score%d.pickle' % i, 'wb')
    pickle.dump(pre_score, fileX, protocol=4)
    fileX.close()

    # 最后做对比图写出来
    #  ######### Print Precision and Recall ##########
    pred_proba = cnn.predict(testX, batch_size=2048)
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
    plt.savefig('./cnn/curve/' + str(type) + 'Precision-Recall%d.png' % i)

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
    plt.savefig('./cnn/curve/' + str(type) + 'ROC %d.png' % i)
    SN, SP = performance(true_class, pred_score)
    pre = precision_score(y_true=true_class, y_pred=pred_score)
    rec = recall_score(y_true=true_class, y_pred=pred_score)
    f1 = f1_score(y_true=true_class, y_pred=pred_score)

    # Sn和recall是同一个值
    return pre_score, pre, rec, SN, SP, f1, mcc, roc_auc


def print_loss(fitHistory, i, type, para):
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
    plt.savefig('./cnn/curve/' + str(type) + str(para) + 'loss%d.png' % i)

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
    plt.savefig('./cnn/curve/' + str(type) + 'loss%d.png' % i)

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


def main(type, numm, para, xlsname, aa):
    if os.path.exists('./cnn/results.txt'):
        print("The result will be written in the 'results.txt' file")
    else:
        with open('./cnn/results.txt', 'w') as file:
            file.write('date' + '\t' + 'type' + '\t' + 'time' + '\t' + 'train loss' + '\t' + 'train acc' + '\t' +
                       'val loss' + '\t' + 'val acc' + '\t' + 'test loss' + '\t' + 'test acc' + '\t' + 'precision' +
                       '\t' + 'recall' + '\t' + 'Sn' + '\t' + 'Sp' + '\t' + 'F1' + '\t' + 'mcc' + '\t' + 'auc' + '\n')
    # if os.path.exists(xlsname):
    #     print("The result will be written in the '%s' file" % xlsname)
    # else:
    #     excel = xlwt.Workbook()
    #     sheet = excel.add_sheet('Sheet1')
    #     sheet.write(0, 0, 'date')
    #     sheet.write(0, 1, 'type')
    #     sheet.write(0, 2, 'train loss')
    #     sheet.write(0, 3, 'train acc')
    #     sheet.write(0, 4, 'val loss')
    #     sheet.write(0, 5, 'val acc')
    #     sheet.write(0, 6, 'test loss')
    #     sheet.write(0, 7, 'test acc')
    #     sheet.write(0, 8, 'precision')
    #     sheet.write(0, 9, 'recall')
    #     sheet.write(0, 10, 'Sn')
    #     sheet.write(0, 11, 'Sp')
    #     sheet.write(0, 12, 'F1')
    #     sheet.write(0, 13, 'mcc')
    #     sheet.write(0, 14, 'auc')
    #     excel.save(xlsname)

    for i in range(1, 2):
        print('time:%d' % i)
        trainX, trainY, input, _ = data_processing.load_data('./m6Adata/trainX%d.pickle' % i)
        valX, valY, _, _ = data_processing.load_data('./m6Adata/trainX%d.pickle' % i)
        # testX, testY, _, _ = data_processing.load_data('./onehot/testX%d.pickle' % i)

        cnn = model(input, numm, aa)
        history = fit_model(trainX, trainY, valX, valY, cnn, i, type)

        loss1, acc1, loss2, acc2 = print_loss(history, i, type, para)
        print('train loss:', loss1,
              'train acc:', acc1,
              'val loss:', loss2,
              'val acc:', acc2)
        pre_score, pre, rec, SN, SP, f1, mcc, roc_auc = evaluate_model('./cnn/model/' + str(type) + '%d_.h5' % i, valX,
                                                                       valY, i, type)
        print('test loss:', pre_score[0],
              'test acc:', pre_score[1])

        with open('./cnn/results.txt', 'a') as file:
            file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\t' + str(type) + '\t' + aa+numm+str(i) + '\t' +
                       str(loss1) + '\t' + str(acc1) + '\t' + str(loss2) + '\t' + str(acc2) + '\t' +
                       str(pre_score[0]) + '\t' + str(pre_score[1]) + '\t' + str(pre) + '\t' + str(rec) + '\t' +
                       str(SN) + '\t' + str(SP) + '\t' + str(f1) + '\t' + str(mcc) + '\t' + str(roc_auc) + '\n')
        # rexcel = open_workbook(xlsname)
        # rows = rexcel.sheets()[0].nrows
        # excel = copy(rexcel)
        # table = excel.get_sheet(0)
        # table.write(rows, 0, 'kernel%s' % numm)
        # table.write(rows, 1, str(i))
        # table.write(rows, 2, str(loss1))
        # table.write(rows, 3, str(acc1))
        # table.write(rows, 4, str(loss2))
        # table.write(rows, 5, str(acc2))
        # table.write(rows, 6, str(pre_score[0]))
        # table.write(rows, 7, str(pre_score[1]))
        # table.write(rows, 8, str(pre))
        # table.write(rows, 9, str(rec))
        # table.write(rows, 10, str(SN))
        # table.write(rows, 11, str(SP))
        # table.write(rows, 12, str(f1))
        # table.write(rows, 13, str(mcc))
        # table.write(rows, 14, str(roc_auc))
        # excel.save(xlsname)


if __name__ == '__main__':
    ''' 
    ret = config.get('global', 'workgroup')
    ret = config.getint('global', 'maxlog')
    ret = config.getfloat('public', 'pi')
    ret = config.getboolean('public', 'public')
    '''

    # for argument in range(1, 21):
    #     argument = argument * 10
    #     type = 'm1A' + str(argument)
    #     main(argument)
    xlsname = 'm6A.xls'
    for i in range(1, 10):
        for a in range(1, 11):
            numm = a * 5
            type = 'm6A'
            para = 'keral %s' % numm
            main(type, numm, para, xlsname, i)
