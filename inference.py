import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from Models import Anti_spoof_net_CNN
from Models import Anti_spoof_net_RNN

warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 5

cnn_model = Anti_spoof_net_CNN.Anti_spoof_net_CNN()
cnn_model.load_state_dict(torch.load('saved_models/cnn_model'))
cnn_model.eval()

rnn_model = Anti_spoof_net_RNN.Anti_spoof_net_RNN()
rnn_model.load_state_dict(torch.load('saved_models/rnn_model_w_pretrained_cnn'))
rnn_model.eval()

one = torch.ones(BATCH_SIZE, 1, 32, 32).to(device)
zero = torch.zeros(BATCH_SIZE, 1, 32, 32).to(device)

threshold = 0.1
CLASS_NAMES = ['Live', 'Spoof']


def imshow_np(img, mode: str, label: str, prob: str):
    timestamp = datetime.now()
    height, width, depth = img.shape

    if depth == 1:
        img = img[:, :, 0]

    fname = 'test_img/' + timestamp.strftime("%d%m%Y_%H%M%S_%f") + '_' + mode + '_' + label + '_' + prob + '.png'
    plt.imsave(fname, img)


def infer(images: np.ndarray):
    images = torch.tensor(np.transpose(images, (0, 3, 1, 2)), dtype=torch.float32)

    hidden = (torch.zeros(1, 1, 100, device=device), torch.zeros(1, 1, 100, device=device))

    with torch.no_grad():
        D, T = cnn_model(images)

        # Non_rigid_registration_layer
        V = torch.where(D >= threshold, one, zero)
        U = T * V
        F = U

        outputs_F, hidden = rnn_model(F, hidden)
        outputs_F = outputs_F.view(BATCH_SIZE, 2)
        outputs_F = nn.Softmax(dim=1)(outputs_F)

        y_pred = outputs_F.cpu().detach().numpy()
        y_pred_class = np.argmax(y_pred, axis=1)
        y_pred_prob = np.max(y_pred, axis=1)

        prediction = list()
        probability = list()

        for i in range(len(images)):
            imshow_np(img=np.transpose(images[i, :, :, :].cpu().numpy(), (1, 2, 0)), mode='raw',
                      label=CLASS_NAMES[y_pred_class[i]], prob=str(y_pred_prob[i]))
            imshow_np(img=np.transpose(D[i, :, :, :].cpu().detach().numpy(), (1, 2, 0)), mode='depth',
                      label=CLASS_NAMES[y_pred_class[i]], prob=str(y_pred_prob[i]))

            prediction.append(CLASS_NAMES[int(y_pred_class[i])])
            probability.append(str(y_pred_prob[i]))

        return prediction, probability
