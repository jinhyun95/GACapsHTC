import numpy as np
import torch


def calc_score(labels, pred):
    true_positive = (labels * (labels == pred)).sum(0)[labels.sum(0) > 0]
    true_negative = ((1. - labels) * (labels == pred)).sum(0)[labels.sum(0) > 0]
    false_positive = ((1. - labels) * (labels != pred)).sum(0)[labels.sum(0) > 0]
    false_negative = (labels * (labels != pred)).sum(0)[labels.sum(0) > 0]
    
    micro_precision = true_positive.sum() / (true_positive.sum() + false_positive.sum())
    micro_recall = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    precisions = true_positive / (true_positive + false_positive + 1e-8)
    recalls = true_positive / (true_positive + false_negative + 1e-8)

    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    macro_f1 = f1s.mean()

    return {'micro-f1': micro_f1, 'macro-f1': macro_f1}

def f1_scores(labels, probs, predictions, label_ids, label_sequences, mode=[0, 0, 0]):
    pred = np.copy(predictions)
    for i in range(labels.shape[0]):
        if mode[0] == 1:
            pred[i, :] = remove_isolated(pred[i, :], probs[i, :], label_ids, label_sequences)
        if mode[0] == 2:
            pred[i, :] = connect_isolated(pred[i, :], probs[i, :], label_ids, label_sequences)
        if mode[1] == 1:
            pred[i, :] = remove_dangling(pred[i, :], probs[i, :], label_ids, label_sequences)
        if mode[2] == 1:
            pred[i, :] = select_argmax_su(pred[i, :], probs[i, :], label_ids, label_sequences)
        if mode[2] == 2:
            pred[i, :] = select_argmax_path(pred[i, :], probs[i, :], label_ids, label_sequences)

    return calc_score(labels, pred)

# post-processing for mandatory leaf assumption
def remove_isolated(prediction, probs, label_ids, paths):
    for path in paths:
        if prediction[path[0]]:
            for topic in path[1:]:
                if not prediction[topic]:
                    prediction[path[0]] = False
                    break
    return prediction

def connect_isolated(prediction, probs, label_ids, paths):
    for path in paths:
        if prediction[path[0]]:
            for topic in path[1:]:
                prediction[topic] = True
    return prediction

def remove_dangling(prediction, probs, label_ids, paths):
    dangling = [True for _ in label_ids]
    for path in paths:
        if prediction[path[0]]:
            for topic in path:
                dangling[topic] = False
    for label in range(len(label_ids)):
        if dangling[label]:
            prediction[label] = False
    return prediction

def select_argmax_su(prediction, probs, label_ids, paths):
    flag = True
    best_p = None
    best_path = []
    for path in paths:
        if prediction[path[0]]:
            flag = False
            break
        else:
            p = probs[path[0]]
            if best_p is None or p > best_p:
                best_p = p
                best_path = path
    if flag:
        for topic in best_path:
            prediction[topic] = True
    return prediction

def select_argmax_path(prediction, probs, label_ids, paths):
    flag = True
    best_p = None
    best_path = []
    for path in paths:
        p = 1
        if prediction[path[0]]:
            flag = False
            break
        else:
            for topic in path:
                p = p * probs[topic]
            p = p ** (1. / len(path))
        if best_p is None or p > best_p:
            best_p = p
            best_path = path
    if flag:
        for topic in best_path:
            prediction[topic] = True
    return prediction

def select_best_path(labels, probs, label_ids, label_sequences):
    if (probs < 0).sum().item() + (probs > 1).sum().item() > 0:
        probs = torch.sigmoid(probs)
    pred = torch.zeros_like(probs).to(torch.int).numpy()
    for i in range(labels.shape[0]):
        pred[i, :] = select_argmax_path(pred[i, :], probs[i, :], label_ids, label_sequences)

    return calc_score(labels, pred)

def select_best_leaf(labels, probs, label_ids, label_sequences):
    pred = torch.zeros_like(probs).to(torch.int).numpy()
    for i in range(labels.shape[0]):
        pred[i, :] = select_argmax_su(pred[i, :], probs[i, :], label_ids, label_sequences)

    return calc_score(labels, pred)

def select_argmax_autoreg(prediction, probs, label_ids, paths):
    best_p = None
    check = None
    for path in paths:
        p = probs[path[1]]
        if best_p is None or p > best_p:
            best_p = p
            check = path[1]
    prediction[check] = True
    
    best_p = None
    check2 = None
    for path in paths:
        if path[1] == check:
            p = probs[path[0]]
            if best_p is None or p > best_p:
                best_p = p
                check2 = path[0]
    prediction[check2] = True
    return prediction

def select_best_path_autoregressive(labels, probs, label_ids, label_sequences):
    pred = torch.zeros_like(probs).to(torch.int).numpy()
    for i in range(labels.shape[0]):
        pred[i, :] = select_argmax_autoreg(pred[i, :], probs[i, :], label_ids, label_sequences)

    return calc_score(labels, pred)
