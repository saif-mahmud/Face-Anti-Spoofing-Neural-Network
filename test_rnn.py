import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

from Models import Anti_spoof_net_RNN


def rnn_test():
    rnn_model = Anti_spoof_net_RNN.Anti_spoof_net_RNN()
    criterion = nn.MSELoss()

    hidden = (torch.zeros(1, 1, 100), torch.zeros(1, 1, 100))

    x = torch.randn(5, 1, 32, 32)
    y, _ = rnn_model(x, hidden)

    norm = torch.linalg.norm(y)
    y = y.view(50)

    print(y)
    print(y.shape)

    print(float(torch.square(norm).detach().numpy()))


def thresh_plot(score_data):
    fig, ax = plt.subplots()
    colors = {0: 'green', 1: 'red'}
    ax.scatter(score_data[:, 0], score_data[:, 1], c=np.vectorize(colors.get)(score_data[:, 2]))

    plt.grid(True)
    plt.xlabel('validation set batch')
    plt.ylabel('score')
    plt.savefig('thresh_plt.png')


def classification_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = score(y_true, y_pred, average='macro')

    return acc, precision, recall, fscore


def tune_hyperparam(score_data, thresh_range=[25000, 200000]):
    clf_score = score_data[:, 1]
    y_true = score_data[:, 2]
    y_pred = np.zeros(y_true.shape)

    hparam_res = list()

    for score_thresh in range(thresh_range[0], thresh_range[1], 5000):
        live_idx = np.where(clf_score < score_thresh)
        y_pred[live_idx] = 1

        acc, precision, recall, fscore = classification_accuracy(y_true, y_pred)
        hparam_res.append(list((score_thresh, acc, precision, recall, fscore)))

    hparam_res = np.array(hparam_res)

    plt.plot(hparam_res[:, 0], hparam_res[:, 1], 'r+-', label='acc')
    plt.plot(hparam_res[:, 0], hparam_res[:, 2], 'gx-', label='prec')
    plt.plot(hparam_res[:, 0], hparam_res[:, 3], 'b^-', label='rec')
    plt.plot(hparam_res[:, 0], hparam_res[:, 4], 'co-', label='f1')

    plt.grid(True)
    plt.legend()
    plt.xlabel('clf. score threshold')
    plt.savefig('score_hparam.png')


if __name__ == '__main__':
    score_data = np.load('thresh_test.npy')

    tune_hyperparam(score_data)
