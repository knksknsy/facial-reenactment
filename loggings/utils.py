import itertools
import os
import codecs
import json
import numpy as np

from sklearn.metrics import auc
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


def plot_roc_curve(filename, fpr, tpr, thresholds, pos_label=1):
    if pos_label == 0:
        tpr, fpr = fpr, tpr

    # Find optimal threshold
    optimal_idx = np.argmax(tpr-fpr)
    threshold = thresholds[optimal_idx].item()
    roc_auc = auc(tpr, fpr) if pos_label > 0 else auc(fpr, tpr)

    size = (500, 500)
    dpi = 100
    figsize = (size[0] / dpi, size[1] / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    lw = 2
        
    plt.plot(fpr, tpr, color='red', lw=lw, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    threshold = f'\nOptimal Threshold={threshold:.4f}' if threshold is not None else ''
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='black', label='Optimal Threshold')
    plt.xlabel(f'False Positive Rate{threshold}')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def save_cm_roc(path:str, epoch: int, cm, fpr, tpr, thresholds, pos_label):
    path = os.path.join(path, 'cm_roc')
    if not os.path.isdir(path):
        os.makedirs(path)

    json_dict = dict()
    json_dict['cm'] = cm.tolist()
    json_dict['roc'] = dict()
    json_dict['roc']['fpr'] = fpr.tolist()
    json_dict['roc']['tpr'] = tpr.tolist()
    json_dict['roc']['thresholds'] = thresholds.tolist()
    json_dict['roc']['pos_label'] = pos_label

    path = os.path.join(path, f'cm_roc_e_{epoch}.json')
    json.dump(json_dict, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=4)


def load_cm_roc(path: str):
    txt = codecs.open(path, 'r', encoding='utf-8').read()
    o = json.loads(txt)

    cm = np.array(o['cm'])
    fpr = np.array(o['roc']['fpr'])
    tpr = np.array(o['roc']['tpr'])
    thresholds = None
    if 'thresholds' in o['roc']:
        thresholds = np.array(o['roc']['thresholds'])
    if 'pos_label' in o['roc']:
        pos_label = o['roc']['pos_label']

    return cm, fpr, tpr, thresholds, pos_label


def plot_prc_curve(filename, precision, recall, threshold, prc_auc, optimal_idx):
    size = (500, 500)
    dpi = 100
    figsize = (size[0] / dpi, size[1] / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    lw = 2
    plt.plot(recall, precision, color='red', lw=lw, label='PR curve (area = %0.2f)' % prc_auc)
    plt.plot([0, 1], [0, 0], color='grey', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, 1.05])
    threshold = f'\nOptimal Threshold={threshold:.2f}' if threshold is not None else ''
    plt.scatter(recall[optimal_idx], precision[optimal_idx], marker='o', color='black', label='Optimal Threshold')
    plt.xlabel(f'Recall{threshold}')
    plt.ylabel('Precision')
    plt.title('Precision-Recall-Curve')
    plt.legend(loc="best")
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def save_prc(path:str, epoch: int, precision, recall, thresholds, threshold, prc_auc):
    path = os.path.join(path, 'prc_curve')
    if not os.path.isdir(path):
        os.makedirs(path)

    json_dict = dict()
    json_dict['prc'] = dict()
    json_dict['prc']['precision'] = precision.tolist()
    json_dict['prc']['recall'] = recall.tolist()
    json_dict['prc']['thresholds'] = thresholds.astype(np.float64).tolist()
    json_dict['prc']['threshold'] = threshold.astype(np.float64)
    json_dict['prc']['prc_auc'] = prc_auc

    path = os.path.join(path, f'prc_e_{epoch}.json')
    json.dump(json_dict, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=4)


def load_prc(path: str):
    txt = codecs.open(path, 'r', encoding='utf-8').read()
    o = json.loads(txt)

    precision = np.array(o['prc']['precision'])
    recall = np.array(o['prc']['recall'])
    thresholds = None
    threshold = None
    if 'threshold' in o['prc']:
        threshold = o['prc']['threshold']
    if 'thresholds' in o['prc']:
        thresholds = np.array(o['prc']['thresholds'])
    prc_auc = o['prc']['prc_auc']

    return precision, recall, thresholds, threshold, prc_auc
