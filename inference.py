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
cnn_model.load_state_dict(torch.load('saved_models/siw/cnn_model'))
cnn_model.to(device)
cnn_model.eval()

rnn_model = Anti_spoof_net_RNN.Anti_spoof_net_RNN()
rnn_model.load_state_dict(torch.load('saved_models/rnn_model_rppg'))
rnn_model.to(device)
rnn_model.eval()

one = torch.ones(BATCH_SIZE, 1, 32, 32, device=device)
zero = torch.zeros(BATCH_SIZE, 1, 32, 32, device=device)

threshold = 0.1  # Non-rigid registration layer depth map threshld
LAMBDA = 0.015  # depth map weight in clf. score
SCORE_THRESHOLD = 10000  # classification score threshold
CLASS_NAMES = {0: 'Live', 1: 'Spoof'}


def imshow_np(img, mode: str, label: str, score: str):
    timestamp = datetime.now()
    height, width, depth = img.shape

    if depth == 1:
        img = img[:, :, 0]

    fname = 'test_img/' + timestamp.strftime("%d%m%Y_%H%M%S_%f") + '_' + mode + '_' + label + '_' + score + '.png'
    plt.imsave(fname, img)

    return fname


def infer(images: np.ndarray):
    images = np.divide(images, 255)
    images = torch.tensor(np.transpose(images, (0, 3, 1, 2)), dtype=torch.float32)
    images = images.to(device)

    hidden = (torch.zeros(1, 1, 100, device=device), torch.zeros(1, 1, 100, device=device))

    with torch.no_grad():
        D, T = cnn_model(images)

        # Non_rigid_registration_layer
        V = torch.where(D >= threshold, one, zero)
        U = T * V
        F = U

        outputs_F, hidden = rnn_model(F, hidden)
        outputs_F = outputs_F.view(50)

        print(D[-1, :, :, :].shape)
        norm_D = torch.linalg.norm(D[-1, :, :, :])
        norm_F = torch.linalg.norm(outputs_F)

        score = torch.square(norm_F) + (LAMBDA * torch.square(norm_D))
        score = score.cpu().detach().numpy()
        y_pred_class = 0 if score > SCORE_THRESHOLD else 1

        prediction = CLASS_NAMES[y_pred_class]

        fname = imshow_np(img=np.transpose(images[-1, :, :, :].cpu().numpy(), (1, 2, 0)), mode='raw',
                          label=prediction, score=str(score))
        _ = imshow_np(img=np.transpose(D[-1, :, :, :].cpu().detach().numpy(), (1, 2, 0)), mode='depth',
                      label=prediction, score=str(score))

        with open('./server_log/result.csv', 'a') as fd:
            csv_row = '\n' + str(fname) + ', ' + str(norm_F.cpu().detach().numpy()) + ', ' + str(norm_D.cpu().detach().numpy())
            fd.write(csv_row)

        return prediction, score
