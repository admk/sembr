import torch
import numpy as np


def binary_metrics(fp, tp, fn, tn=None, prefix=None):
    def safe_div(a, b):
        return a / b if b > 0 else 0
    result = {
        'precision': safe_div(tp, tp + fp),
        'recall': safe_div(tp, tp + fn),
        'f1': safe_div(2 * tp, 2 * tp + fp + fn),
        'iou': safe_div(tp, tp + fp + fn),
    }
    if tn is not None:
        accuracies = {
            'accuracy': safe_div(tp + tn, tn + fn + tp + fp),
            'balanced_accuracy': (
                safe_div(tp, tp + fn) + safe_div(tn, tn + fp)) / 2,
        }
        result.update(accuracies)
    if prefix is not None:
        result = {f'{prefix}_{k}': v for k, v in result.items()}
    return result


def compute_metrics(result):
    logits, labels = result
    preds = np.argmax(logits, axis=2)
    idx = labels != -100
    logits, preds, labels = logits[idx], preds[idx], labels[idx]
    loss = torch.nn.functional.cross_entropy(
        torch.Tensor(logits), torch.LongTensor(labels), reduction='mean')
    preds_set = set(preds.nonzero()[0])
    labels_set = set(labels.nonzero()[0])
    fp = len(preds_set - labels_set)
    tp = len(labels_set & preds_set)
    fn = len(labels_set - preds_set)
    tn = len(set(range(len(preds))) - (labels_set | preds_set))
    metrics = {
        **binary_metrics(fp, tp, fn, tn),
        'overall_accuracy': (preds == labels).mean(),
        'loss': loss,
    }
    return metrics
