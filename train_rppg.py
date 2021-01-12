import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Models import Anti_spoof_net_CNN
from Models import Anti_spoof_net_RNN
from data_processing.rppg_data_generator import get_rppg_dataloader

warnings.filterwarnings("ignore")

print('GPU count:', torch.cuda.device_count())
gpu0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 5
LAMBDA = 0.015


def imshow_np(i, img, mode: str):
    height, width, depth = img.shape

    if depth == 1:
        img = img[:, :, 0]

    fname = 'sample_img/' + str(i) + mode + '.png'
    plt.imsave(fname, img)


def thresh_plot(score_data):
    fig, ax = plt.subplots()
    colors = {0: 'green', 1: 'red'}
    ax.scatter(score_data[:, 0], score_data[:, 1], c=np.vectorize(colors.get)(score_data[:, 2]))

    plt.grid(True)
    plt.savefig('thresh_plt.png')


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

    plt.plot(hparam_res[:, 0], hparam_res[:, 1], 'ro-', label='acc')
    plt.plot(hparam_res[:, 0], hparam_res[:, 2], 'gx-', label='prec')
    plt.plot(hparam_res[:, 0], hparam_res[:, 3], 'b^-', label='rec')
    plt.plot(hparam_res[:, 0], hparam_res[:, 4], 'k+-', label='f1')

    plt.grid(True)
    plt.legend()
    plt.xlabel('clf. score threshold')
    plt.savefig('score_hparam.png')


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def prepare_dataloader_D(data_images_train, data_images_test, data_labels_D_train, data_labels_D_test):
    trainset_D = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data_images_train, (0, 3, 1, 2))),
                                                torch.tensor(data_labels_D_train))
    testset_D = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data_images_test, (0, 3, 1, 2))),
                                               torch.tensor(data_labels_D_test))

    trainloader_D = torch.utils.data.DataLoader(trainset_D, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    testloader_D = torch.utils.data.DataLoader(testset_D, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    return trainloader_D, testloader_D


def get_dataloader():
    Images = np.load('data_processing/siw_npz/images.npz')
    Labels_D = np.load('data_processing/siw_npz/labels_D.npz')
    Labels = np.load('data_processing/siw_npz/label.npz')

    n = len(Images)

    data_images = np.zeros((n, 256, 256, 3), dtype=np.float32)
    data_labels_D = np.zeros((n, 32, 32, 1), dtype=np.float32)
    data_labels = np.zeros((n), dtype=np.float32)

    for item in Images.files:
        data_images[int(item), :, :, :] = Images[item]
        data_labels_D[int(item), :, :, :] = Labels_D[item]
        data_labels[int(item)] = Labels[item]

    # print(data_images.shape)
    # print(data_labels_D.shape)
    # print(data_labels.shape)

    data_images_train, data_images_test, data_labels_train, data_labels_test = train_test_split(data_images,
                                                                                                data_labels,
                                                                                                test_size=0.20,
                                                                                                random_state=42)
    data_images_train, data_images_test, data_labels_D_train, data_labels_D_test = train_test_split(data_images,
                                                                                                    data_labels_D,
                                                                                                    test_size=0.20,
                                                                                                    random_state=42)

    # print(data_labels_train.shape)
    # print(data_labels_train.shape)
    # print(data_labels_D_train.shape)

    # print(data_images_test.shape)
    # print(data_labels_test.shape)
    # print(data_labels_D_test.shape)

    # unique, counts = np.unique(data_labels_train, return_counts=True)
    # print(unique, counts)

    # unique, counts = np.unique(data_labels_test, return_counts=True)
    # print(unique, counts)

    trainloader_D, testloader_D = prepare_dataloader_D(data_images_train, data_images_test, data_labels_D_train,
                                                       data_labels_D_test)

    return trainloader_D, testloader_D, data_labels_train, data_labels_test


def train_CNN(cnn_model, optimizer, trainloader, criterion, cnn_device, n_epoch=10):
    cnn_model = cnn_model.to(cnn_device)

    for epoch in range(n_epoch):

        print('\n[CNN Epoch: %2d / %2d]' % (epoch + 1, n_epoch))

        running_loss = 0.0
        total = 0

        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            images, labels_D = data
            images, labels_D = images.to(cnn_device), labels_D.to(cnn_device)

            optimizer.zero_grad()

            outputs_D, _ = cnn_model(images)

            # handle NaN:
            if torch.norm((outputs_D != outputs_D).float()) == 0:
                if epoch == (n_epoch - 1):
                    imshow_np(i, np.transpose(images[-1, :, :, :].cpu().numpy(), (1, 2, 0)), mode='_raw')
                    imshow_np(i, np.transpose(outputs_D[-1, :, :, :].cpu().detach().numpy(), (1, 2, 0)),
                              mode=('_depth_epoch_' + str(epoch + 1)))

                loss = criterion(outputs_D, labels_D)

                loss.backward()
                optimizer.step()

                total += labels_D.size(0)
                running_loss += loss.item()

                norm_D = torch.linalg.norm(outputs_D[-1, :, :, :])
                with open('./training_log/depth_norm.csv', 'a') as fd:
                    csv_row = '\n' + str(epoch) + ', ' + str(i) + ', ' + str(norm_D.cpu().detach().numpy())
                    fd.write(csv_row)

        print('[CNN: Epoch: %d - MSE loss on depth maps: %.3f]' % (epoch + 1, running_loss / total))

        # model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': cnn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': (running_loss / total),
        }, 'checkpoints/siw/cnn.pt')

    print('CNN: Finished Training')
    torch.save(cnn_model.state_dict(), 'saved_models/siw/cnn_model')


def train_RNN(rnn_model, pretrained_cnn, trainloader, rppg_label, optimizer, criterion, cnn_device, rnn_device,
              n_epoch=10):
    threshold = 0.1

    pretrained_cnn = pretrained_cnn.to(cnn_device)
    rnn_model = rnn_model.to(rnn_device)

    rppg_label = rppg_label.to(rnn_device)

    one = torch.ones(BATCH_SIZE, 1, 32, 32).to(cnn_device)
    zero = torch.zeros(BATCH_SIZE, 1, 32, 32).to(cnn_device)

    hidden = (torch.zeros(1, 1, 100, device=rnn_device), torch.zeros(1, 1, 100, device=rnn_device))

    for epoch in range(n_epoch):

        print('\n[RNN: Epoch: %2d / %2d]' % (epoch + 1, n_epoch))

        running_loss = 0.0

        for i, images in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            images = images[0].to(cnn_device)

            with torch.no_grad():
                D, T = pretrained_cnn(images)

            # Non_rigid_registration_layer
            V = torch.where(D >= threshold, one, zero)
            U = T * V
            F = U

            F = F.to(rnn_device)

            optimizer.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs_F, hidden = rnn_model(F, hidden)
            outputs_F = outputs_F.view(50)

            # handle NaN:
            if torch.norm((outputs_F != outputs_F).float()) == 0:
                loss = criterion(outputs_F, rppg_label[i])
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        print('[Epoch: %d - rppg loss: %.5f' % (epoch + 1, running_loss / len(trainloader)))

    print('RNN: Finished Training')
    torch.save(rnn_model.state_dict(), 'saved_models/rnn_model_rppg')


def classification_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = score(y_true, y_pred, average='macro')

    return acc, precision, recall, fscore


def predict(testloader, spoof_labels, cnn_model, rnn_model, device):
    threshold = 0.1

    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    rnn_model = rnn_model.to(device)
    rnn_model.eval()

    one = torch.ones(BATCH_SIZE, 1, 32, 32).to(device)
    zero = torch.zeros(BATCH_SIZE, 1, 32, 32).to(device)

    hidden = (torch.zeros(1, 1, 100, device=device), torch.zeros(1, 1, 100, device=device))

    # Y_TEST = list()
    # Y_PRED = list()
    score_data = list()

    for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
        images, labels_D = data
        images, labels_D = images.to(device), labels_D.to(device)

        with torch.no_grad():
            D, T = cnn_model(images)

            # Non_rigid_registration_layer
            V = torch.where(D >= threshold, one, zero)
            U = T * V
            F = U

            hidden = repackage_hidden(hidden)
            outputs_F, hidden = rnn_model(F, hidden)
            outputs_F = outputs_F.view(50)

            label = spoof_labels[(i * BATCH_SIZE):(i * BATCH_SIZE) + BATCH_SIZE][-1]

            # print(D.shape)

            norm_D = torch.linalg.norm(D[-1, :, :, :])
            norm_F = torch.linalg.norm(outputs_F)

            score = torch.square(norm_F) + (LAMBDA * torch.square(norm_D))
            print('clf. score: %.5f | label: %d' % (score, label))

            score_data.append(list((i, float(score.cpu().detach().numpy()), float(label))))

            # Y_TEST.extend(label.cpu().detach().numpy())
            # Y_PRED.extend(outputs_F.cpu().detach().numpy())

    # acc, precision, recall, fscore = classification_accuracy(Y_TEST, Y_PRED)
    # print('[Prediction: acc: %.5f | prec: %.5f | rec: %.5f | f1: %.5f]' % (acc, precision, recall, fscore))
    print(score_data)
    np.save('thresh_test.npy', score_data)
    thresh_plot(np.array(score_data))


if __name__ == '__main__':
    trainloader_D, testloader_D, data_labels_train, data_labels_test = get_dataloader()

    cnn_model = Anti_spoof_net_CNN.Anti_spoof_net_CNN()
    rnn_model = Anti_spoof_net_RNN.Anti_spoof_net_RNN()

    criterion = nn.MSELoss()

    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

    if str(sys.argv[1]) == 'cnn':
        train_CNN(cnn_model=cnn_model, optimizer=cnn_optimizer, trainloader=trainloader_D, criterion=criterion,
                  cnn_device=gpu1, n_epoch=75)

    elif str(sys.argv[1]) == 'rnn':
        img_dataloader, rppg_data = get_rppg_dataloader()
        cnn_model.load_state_dict(torch.load('saved_models/cnn_model'))
        cnn_model.eval()

        train_RNN(rnn_model=rnn_model, pretrained_cnn=cnn_model, optimizer=rnn_optimizer, trainloader=img_dataloader,
                  rppg_label=rppg_data, criterion=criterion, cnn_device=gpu1, rnn_device=gpu1, n_epoch=40)

    elif str(sys.argv[1]) == 'pred':
        cnn_model.load_state_dict(torch.load('saved_models/cnn_model'))
        rnn_model.load_state_dict(torch.load('saved_models/rnn_model_rppg'))

        predict(testloader=testloader_D, spoof_labels=data_labels_test, cnn_model=cnn_model, rnn_model=rnn_model,
                device=gpu1)
