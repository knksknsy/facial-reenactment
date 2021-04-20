import itertools
import os
import codecs
import json
import numpy as np

import matplotlib.pyplot as plt

def plot_confusion_matrix(filename, cm):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    size = (500, 500)
    cmap = plt.get_cmap('Greys')

    dpi = 100
    figsize = (size[0] / dpi, size[1] / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()

    target_names = ['fake', 'real']
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:,}'.format(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'predicted label\naccuracy={accuracy:.4f}; misclass={misclass:.4f}')
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curve(filename, fpr, tpr, threshold, roc_auc):
    size = (500, 500)
    dpi = 100
    figsize = (size[0] / dpi, size[1] / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    lw = 2
    plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    threshold = f'\nOptimal Threshold={threshold:.2f}' if threshold is not None else ''
    plt.xlabel(f'False Positive Rate{threshold}')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def save_cm_roc(path:str, epoch: int, cm, fpr, tpr, threshold, roc_auc):
    path = os.path.join(path, 'cm_roc')
    if not os.path.isdir(path):
        os.makedirs(path)

    json_dict = dict()
    json_dict['cm'] = cm.tolist()
    json_dict['roc'] = dict()
    json_dict['roc']['fpr'] = fpr.tolist()
    json_dict['roc']['tpr'] = tpr.tolist()
    json_dict['roc']['threshold'] = threshold
    json_dict['roc']['roc_auc'] = roc_auc

    path = os.path.join(path, f'cm_roc_e_{epoch}.json')
    json.dump(json_dict, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=4)


def load_cm_roc(path: str):
    txt = codecs.open(path, 'r', encoding='utf-8').read()
    o = json.loads(txt)

    cm = np.array(o['cm'])
    fpr = np.array(o['roc']['fpr'])
    tpr = np.array(o['roc']['tpr'])
    threshold = None
    if 'threshold' in o['roc']:
        threshold = o['roc']['threshold']
    roc_auc = o['roc']['roc_auc']

    return cm, fpr, tpr, threshold, roc_auc
