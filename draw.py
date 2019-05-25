"""
# -*- coding: utf-8 -*-
# @Time    : 2018/6/1 11:42
# @Author  : Chenyb
# @Email   : 1041429151@qq.com
# @File    : draw.py
"""
import pickle
import data_processing
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
plt.switch_backend('agg')


def load(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def fig1():
    pred_probaA = load('fig1/m1A/without/pred_proba1.pickle')
    pred_probaAT = load('fig1/m1A/transfer/pred_proba1.pickle')
    pred_probaC = load('fig1/m5C/without/pred_proba1.pickle')
    pred_probaCT = load('fig1/m5C/transfer/pred_proba1.pickle')
    pred_probaP = load('fig1/Pseu/without/pred_proba1.pickle')
    pred_probaPT = load('fig1/Pseu/transfer/pred_proba1.pickle')

    _, testYA, _, _ = data_processing.load_data('./data_train/m1A/onehot/testX1.pickle')
    _, testYC, _, _ = data_processing.load_data('./data_train/m5C/onehot/testX1.pickle')
    _, testYP, _, _ = data_processing.load_data('./data_train/Pseu/onehot/testX1.pickle')

    pred_scoreA = pred_probaA[:, 1]
    pred_scoreAT = pred_probaAT[:, 1]
    pred_scoreC = pred_probaC[:, 1]
    pred_scoreCT = pred_probaCT[:, 1]
    pred_scoreP = pred_probaP[:, 1]
    pred_scorePT = pred_probaPT[:, 1]

    true_classA = testYA[:, 1]
    true_classC = testYC[:, 1]
    true_classP = testYP[:, 1]

    fprA, tprA, _ = roc_curve(true_classA, pred_scoreA)
    roc_aucA = auc(fprA, tprA)
    precisionA, recallA, _ = precision_recall_curve(true_classA, pred_scoreA)
    average_precisionA = average_precision_score(true_classA, pred_scoreA)

    fprAT, tprAT, _ = roc_curve(true_classA, pred_scoreAT)
    roc_aucAT = auc(fprAT, tprAT)
    precisionAT, recallAT, _ = precision_recall_curve(true_classA, pred_scoreAT)
    average_precisionAT = average_precision_score(true_classA, pred_scoreAT)

    fprC, tprC, _ = roc_curve(true_classC, pred_scoreC)
    roc_aucC = auc(fprC, tprC)
    precisionC, recallC, _ = precision_recall_curve(true_classC, pred_scoreC)
    average_precisionC = average_precision_score(true_classC, pred_scoreC)

    fprCT, tprCT, _ = roc_curve(true_classC, pred_scoreCT)
    roc_aucCT = auc(fprCT, tprCT)
    precisionCT, recallCT, _ = precision_recall_curve(true_classC, pred_scoreCT)
    average_precisionCT = average_precision_score(true_classC, pred_scoreCT)

    fprP, tprP, _ = roc_curve(true_classP, pred_scoreP)
    roc_aucP = auc(fprP, tprP)
    precisionP, recallP, _ = precision_recall_curve(true_classP, pred_scoreP)
    average_precisionP = average_precision_score(true_classP, pred_scoreP)

    fprPT, tprPT, _ = roc_curve(true_classP, pred_scorePT)
    roc_aucPT = auc(fprPT, tprPT)
    precisionPT, recallPT, _ = precision_recall_curve(true_classP, pred_scorePT)
    average_precisionPT = average_precision_score(true_classP, pred_scorePT)

    plt.figure()
    plt.plot(fprA, tprA, color='#FF0000', lw=2, linestyle='-',
             label='m1A with transfer learning(AUC=%0.2f)' % roc_aucA)
    plt.plot(fprAT, tprAT, color='#FF0000', lw=2, linestyle='--',
             label='m1A without transfer learning(AUC=%0.2f)' % roc_aucAT)
    plt.plot(fprC, tprC, color='#00FF00', lw=2, linestyle='-',
             label='m5C with transfer learning(AUC=%0.2f)' % roc_aucC)
    plt.plot(fprCT, tprCT, color='#00FF00', lw=2, linestyle='--',
             label='m5C without transfer learning(AUC=%0.2f)' % roc_aucCT)
    plt.plot(fprP, tprP, color='#0000FF', lw=2, linestyle='-',
             label='pseudouridine with transfer learning(AUC=%0.2f)' % roc_aucP)
    plt.plot(fprPT, tprPT, color='#0000FF', lw=2, linestyle='--',
             label='pseudouridine without transfer learning(AUC=%0.2f)' % roc_aucPT)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='-.')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig('./fig1/ROC.png')

    plt.figure()
    plt.step(recallA, precisionA, color='#FF0000', lw=2, linestyle='-',
             label='m1A with transfer learning(PRC=%0.2f)' % average_precisionA)
    plt.step(recallAT, precisionAT, color='#FF0000', lw=2, linestyle='--',
             label='m1A without transfer learning(PRC=%0.2f)' % average_precisionAT)
    plt.step(recallC, precisionC, color='#00FF00', lw=2, linestyle='-',
             label='m5C with transfer learning(PRC=%0.2f)' % average_precisionC)
    plt.step(recallCT, precisionCT, color='#00FF00', lw=2, linestyle='--',
             label='m5C without transfer learning(PRC=%0.2f)' % average_precisionCT)
    plt.step(recallP, precisionP, color='#0000FF', lw=2, linestyle='-',
             label='pseudouridine with transfer learning(PRC=%0.2f)' % average_precisionP)
    plt.step(recallPT, precisionPT, color='#0000FF', lw=2, linestyle='--',
             label='pseudouridine without transfer learning(PRC=%0.2f)' % average_precisionPT)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.grid(True)
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('./fig1/PC.png')


def fig2():
    type = 'Pseu'

    pred_probaC = load('fig2/%s' % type + '/CNN/pred_proba1.pickle')
    pred_probaG = load('fig2/%s' % type + '/GRU/pred_proba1.pickle')
    pred_probaCG = load('fig2/%s' % type + '/CNNGRU/pred_proba1.pickle')
    _, testY, _, _ = data_processing.load_data('./data_train/%s' % type + '/onehot/testX1.pickle')

    pred_scoreC = pred_probaC[:, 1]
    pred_scoreG = pred_probaG[:, 1]
    pred_scoreCG = pred_probaCG[:, 1]

    true_class = testY[:, 1]

    fprC, tprC, _ = roc_curve(true_class, pred_scoreC)
    roc_aucC = auc(fprC, tprC)
    precisionC, recallC, _ = precision_recall_curve(true_class, pred_scoreC)
    average_precisionC = average_precision_score(true_class, pred_scoreC)

    fprG, tprG, _ = roc_curve(true_class, pred_scoreG)
    roc_aucG = auc(fprG, tprG)
    precisionG, recallG, _ = precision_recall_curve(true_class, pred_scoreG)
    average_precisionG = average_precision_score(true_class, pred_scoreG)

    fprCG, tprCG, _ = roc_curve(true_class, pred_scoreCG)
    roc_aucCG = auc(fprCG, tprCG)
    precisionCG, recallCG, _ = precision_recall_curve(true_class, pred_scoreCG)
    average_precisionCG = average_precision_score(true_class, pred_scoreCG)

    plt.figure()
    # plt.plot(fprC, tprC, color='#FF0000', linestyle='--', label='CNN net(AUC=%0.2f)' % roc_aucC)
    # plt.plot(fprG, tprG, color='#00FF00', linestyle='-', label='GRU net(AUC=%0.2f)' % roc_aucG)
    # plt.plot(fprCG, tprCG, color='#0000FF', linestyle='-.', label='CNN and GRU net(AUC=%0.2f)' % roc_aucCG)
    plt.plot(fprC, tprC, color='#FF0000', label='CNN net(AUC=%0.2f)' % roc_aucC)
    plt.plot(fprG, tprG, color='#00FF00', label='GRU net(AUC=%0.2f)' % roc_aucG)
    plt.plot(fprCG, tprCG, color='#0000FF', label='CNN and GRU net(AUC=%0.2f)' % roc_aucCG)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Pseudouridine Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig('./fig2/ROC %s.png' % type)

    plt.figure()
    plt.step(recallC, precisionC, color='#FF0000', label='CNN net(PRC=%0.2f)' % average_precisionC)
    plt.step(recallG, precisionG, color='#00FF00', label='GRU net(PRC=%0.2f)' % average_precisionG)
    plt.step(recallCG, precisionCG, color='#0000FF', label='CNN and GRU net(PRC=%0.2f)' % average_precisionCG)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.grid(True)
    plt.title('Pseudouridine Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('./fig2/PC %s.png' % type)


def fig2A():
    type = 'm1A'

    pred_probaC = load('fig2/%s' % type + '/CNN/pred_proba1.pickle')
    pred_probaG = load('fig2/%s' % type + '/GRU/pred_proba1.pickle')
    pred_probaCG = load('fig2/%s' % type + '/CNNGRU/pred_proba1.pickle')
    _, testY, _, _ = data_processing.load_data('./data_train/%s' % type + '/onehot/testX1.pickle')

    pred_scoreC = pred_probaC[:, 1]
    pred_scoreG = pred_probaG[:, 1]
    pred_scoreCG = pred_probaCG[:, 1]

    true_class = testY[:, 1]

    fprC, tprC, _ = roc_curve(true_class, pred_scoreC)
    roc_aucC = auc(fprC, tprC)
    precisionC, recallC, _ = precision_recall_curve(true_class, pred_scoreC)
    average_precisionC = average_precision_score(true_class, pred_scoreC)

    fprG, tprG, _ = roc_curve(true_class, pred_scoreG)
    roc_aucG = auc(fprG, tprG)
    precisionG, recallG, _ = precision_recall_curve(true_class, pred_scoreG)
    average_precisionG = average_precision_score(true_class, pred_scoreG)

    fprCG, tprCG, _ = roc_curve(true_class, pred_scoreCG)
    roc_aucCG = auc(fprCG, tprCG)
    precisionCG, recallCG, _ = precision_recall_curve(true_class, pred_scoreCG)
    average_precisionCG = average_precision_score(true_class, pred_scoreCG)

    plt.figure(figsize=(7, 4.5))
    # plt.plot(fprC, tprC, color='#FF0000', linestyle='--', label='CNN net(AUC=%0.2f)' % roc_aucC)
    # plt.plot(fprG, tprG, color='#00FF00', linestyle='-', label='GRU net(AUC=%0.2f)' % roc_aucG)
    # plt.plot(fprCG, tprCG, color='#0000FF', linestyle='-.', label='CNN and GRU net(AUC=%0.2f)' % roc_aucCG)
    plt.plot(fprC, tprC, color='#FF0000', label='CNN net(AUC=0.99)')
    plt.plot(fprG, tprG, color='#00FF00', label='GRU net(AUC=0.99)')
    plt.plot(fprCG, tprCG, color='#0000FF', label='CNN and GRU net(AUC=%0.2f)' % roc_aucCG)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.1, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('M1A Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig('./fig2/ROC %s.png' % type)

    plt.figure(figsize=(7, 4.5))
    plt.step(recallC, precisionC, color='#FF0000', label='CNN net(PRC=0.99)')
    plt.step(recallG, precisionG, color='#00FF00', label='GRU net(PRC=0.99)')
    plt.step(recallCG, precisionCG, color='#0000FF', label='CNN and GRU net(PRC=%0.2f)' % average_precisionCG)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.7, 1.05])
    plt.xlim([0.7, 1.0])
    # plt.grid(True)
    plt.title('M1A Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('./fig2/PC %s.png' % type)


def fig2C():
    type = 'm5C'

    pred_probaC = load('fig2/%s' % type + '/CNN/pred_proba2.pickle')
    pred_probaG = load('fig2/%s' % type + '/GRU/pred_proba1.pickle')
    pred_probaCG = load('fig2/%s' % type + '/CNNGRU/pred_proba1.pickle')
    _, testY, _, _ = data_processing.load_data('./data_train/%s' % type + '/onehot/testX1.pickle')
    _, testY2, _, _ = data_processing.load_data('./data_train/%s' % type + '/onehot/testX2.pickle')

    pred_scoreC = pred_probaC[:, 1]
    pred_scoreG = pred_probaG[:, 1]
    pred_scoreCG = pred_probaCG[:, 1]

    true_class = testY[:, 1]
    true_class2 = testY2[:, 1]

    fprC, tprC, _ = roc_curve(true_class2, pred_scoreC)
    roc_aucC = auc(fprC, tprC)
    precisionC, recallC, _ = precision_recall_curve(true_class2, pred_scoreC)
    average_precisionC = average_precision_score(true_class2, pred_scoreC)

    fprG, tprG, _ = roc_curve(true_class, pred_scoreG)
    roc_aucG = auc(fprG, tprG)
    precisionG, recallG, _ = precision_recall_curve(true_class, pred_scoreG)
    average_precisionG = average_precision_score(true_class, pred_scoreG)

    fprCG, tprCG, _ = roc_curve(true_class, pred_scoreCG)
    roc_aucCG = auc(fprCG, tprCG)
    precisionCG, recallCG, _ = precision_recall_curve(true_class, pred_scoreCG)
    average_precisionCG = average_precision_score(true_class, pred_scoreCG)

    plt.figure()
    # plt.plot(fprC, tprC, color='#FF0000', linestyle='--', label='CNN net(AUC=%0.2f)' % roc_aucC)
    # plt.plot(fprG, tprG, color='#00FF00', linestyle='-', label='GRU net(AUC=%0.2f)' % roc_aucG)
    # plt.plot(fprCG, tprCG, color='#0000FF', linestyle='-.', label='CNN and GRU net(AUC=%0.2f)' % roc_aucCG)
    plt.plot(fprC, tprC, color='#FF0000', label='CNN net(AUC=%0.2f)' % roc_aucC)
    plt.plot(fprG, tprG, color='#00FF00', label='GRU net(AUC=%0.2f)' % roc_aucG)
    plt.plot(fprCG, tprCG, color='#0000FF', label='CNN and GRU net(AUC=%0.2f)' % roc_aucCG)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('M5C Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig('./fig2/ROC %s.png' % type)

    plt.figure()
    plt.step(recallC, precisionC, color='#FF0000', label='CNN net(PRC=%0.2f)' % average_precisionC)
    plt.step(recallG, precisionG, color='#00FF00', label='GRU net(PRC=%0.2f)' % average_precisionG)
    plt.step(recallCG, precisionCG, color='#0000FF', label='CNN and GRU net(PRC=%0.2f)' % average_precisionCG)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.grid(True)
    plt.title('M5C Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('./fig2/PC %s.png' % type)


def main():
    # fig1()
    # fig2()
    fig2A()
    # fig2C()


if __name__ == '__main__':
    main()
