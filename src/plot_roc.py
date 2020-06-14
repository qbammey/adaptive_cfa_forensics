#!/usr/bin/env python3

import sys
import argparse

import numpy as np
from matplotlib import pyplot as plt
#import seaborn as sns

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def get_parser():
    parser = argparse.ArgumentParser(description="Plot ROC curves for multiple results", epilog='USAGE:\n\tpython3 plot_roc.py "Label 1" roc1.npz "Label 2" roc2.npz …', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs='+', type=str)
    return parser


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    iterator = iter(args.input)
    rocs, labels = [], []
    for i in iterator:
        labels.append(i)
        rocs.append(next(iterator))

    fpr_global, tpr_global, auc_global, fpr_local, tpr_local, auc_local = [], [], [], [], [], []
    for roc in rocs:
        res = np.load(roc)
        fpr_global.append(res['fpr_global'])
        tpr_global.append(res['tpr_global'])
        auc_global.append(res['auc_global'])
        fpr_local.append(res['fpr_local'])
        tpr_local.append(res['tpr_local'])
        auc_local.append(res['auc_local'])

    plt.plot(np.linspace(0, 1, 300), np.linspace(0, 1, 300), '--', lw=1)
    for fpr, tpr, auc, label in zip(fpr_global, tpr_global, auc_global, labels):
        plt.plot(fpr, tpr, label=f'{label}, AUC: {auc}', lw=3)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Global results')
    plt.show()

    
    plt.plot(np.linspace(0, 1, 300), np.linspace(0, 1, 300), '--', lw=1)
    for fpr, tpr, auc, label in zip(fpr_local, tpr_local, auc_local, labels):
        plt.plot(fpr, tpr, label=f'{label}, AUC: {auc}', lw=3)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Local results')
    plt.show()
    
