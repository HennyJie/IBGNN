from sklearn.metrics import confusion_matrix
import sklearn
import numpy as np


def cal_metrics(gt_label, predicted_label):
    fpr, tpr, _ = sklearn.metrics.roc_curve(gt_label, predicted_label)
    auc = 100 * sklearn.metrics.auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(gt_label, predicted_label).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fp_rate = fp / (fp + tn)
    fn_rate = fn / (fn + tp)

    sum_correct = 0
    num_labels = np.asarray(predicted_label).size

    for i in range(num_labels):
        gt_i = gt_label[i]
        predicted_i = predicted_label[i]
        if gt_i == predicted_i:
            sum_correct += 1

    accuracy = sum_correct / num_labels
    accuracy_macro = sklearn.metrics.f1_score(predicted_label, gt_label, average='macro')

    return accuracy, auc, accuracy_macro, \
           specificity * 100, fp_rate * 100, fn_rate * 100, sensitivity * 100
