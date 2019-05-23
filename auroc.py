from sklearn.metrics import roc_curve, auc
import os
from utils import load_label
import matplotlib.pyplot as plt

def auroc(score, label_path):
    label_dir = 'test_label'

    labels, cnt = load_label(label_dir, True)
    assert len(score) <= cnt, "length of score(%d) is larger than labels(%d)'" % (len(score), cnt)
    labels = labels[:len(score)]
    fpr, tpr, threshold = roc_curve(labels, score)
    acc = auc(fpr, tpr)
    return fpr, tpr, threshold, acc

def plot_roc(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.show()