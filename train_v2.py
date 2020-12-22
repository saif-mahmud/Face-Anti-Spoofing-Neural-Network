import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm

from Models import Anti_spoof_net_CNN
from Models import Anti_spoof_net_RNN

warnings.filterwarnings("ignore")

print('GPU count:', torch.cuda.device_count())
gpu0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def imshow_np(i, img, mode: str):
    height, width, depth = img.shape

    if depth == 1:
        img = img[:, :, 0]

    fname = 'sample_img/' + str(i) + mode + '.png'
    plt.imsave(fname, img)


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

    trainloader_D = torch.utils.data.DataLoader(trainset_D, batch_size=5, shuffle=False, drop_last=True)
    testloader_D = torch.utils.data.DataLoader(testset_D, batch_size=5, shuffle=False, drop_last=True)

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

    training_part = 45 / 55
    n_train = int(n * training_part)

    # Training set
    data_images_train = data_images[:n_train, :, :, :]
    data_labels_D_train = data_labels_D[:n_train, :, :, :]
    data_labels_train = data_labels[:n_train]

    # Test set
    data_images_test = data_images[n_train:, :, :, :]
    data_labels_D_test = data_labels_D[n_train:, :, :, :]
    data_labels_test = data_labels[n_train:]

    trainloader_D, testloader_D = prepare_dataloader_D(data_images_train, data_images_test, data_labels_D_train,
                                                       data_labels_D_test)

    return trainloader_D, testloader_D, data_labels_train, data_labels_test


def train_CNN(cnn_model, optimizer, trainloader, criterion, cnn_device, n_epoch=10):
    cnn_model = cnn_model.to(cnn_device)

    for epoch in range(n_epoch):

        print('\n[CNN Epoch: %2d / %2d]' % (epoch + 1, n_epoch))

        running_loss = 0.0
        total = 0

        for i, data in tqdm(enumerate(trainloader, 0)):
            images, labels_D = data
            images, labels_D = images.to(cnn_device), labels_D.to(cnn_device)

            optimizer.zero_grad()

            outputs_D, _ = cnn_model(images)

            # handle NaN:
            if torch.norm((outputs_D != outputs_D).float()) == 0:
                if epoch == (n_epoch - 1):
                    imshow_np(i, np.transpose(images[0, :, :, :].cpu().numpy(), (1, 2, 0)), mode='_raw')
                    imshow_np(i, np.transpose(outputs_D[0, :, :, :].cpu().detach().numpy(), (1, 2, 0)),
                              mode=('_depth' + str(epoch)))

                loss = criterion(outputs_D, labels_D)

                loss.backward()
                optimizer.step()

                total += labels_D.size(0)
                running_loss += loss.item()

        print('[CNN: Epoch: %d - loss: %.3f]' % (epoch + 1, running_loss / total))

    print('CNN: Finished Training')
    torch.save(cnn_model.state_dict(), 'saved_models/cnn_model')


def train_RNN(rnn_model, pretrained_cnn, optimizer, trainloader, spoof_labels, criterion, cnn_device, rnn_device,
              n_epoch=10):
    threshold = 0.1

    pretrained_cnn = pretrained_cnn.to(cnn_device)
    rnn_model = rnn_model.to(rnn_device)

    one = torch.ones(5, 1, 32, 32).to(cnn_device)
    zero = torch.zeros(5, 1, 32, 32).to(cnn_device)

    hidden = (torch.zeros(1, 1, 100, device=rnn_device),
              torch.zeros(1, 1, 100, device=rnn_device))

    Y_TEST = list()
    Y_PRED = list()

    for epoch in range(n_epoch):

        print('\n[RNN: Epoch: %2d / %2d]' % (epoch + 1, n_epoch))

        running_loss = 0.0

        for i, data in tqdm(enumerate(trainloader, 0)):
            images, labels_D = data
            images = images.to(cnn_device)

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

            # handle NaN:
            if torch.norm((outputs_F != outputs_F).float()) == 0:
                label = spoof_labels[(i * 5): (i * 5) + 5]

                label = torch.from_numpy(label)
                label = label.to(rnn_device)

                outputs_F = outputs_F.view(5, 2)

                loss = criterion(outputs_F, label.long())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                outputs_F = nn.Softmax(dim=1)(outputs_F)
                Y_TEST.extend(label.cpu().detach().numpy())
                Y_PRED.extend(outputs_F.cpu().detach().numpy())

        acc, precision, recall, fscore = classification_accuracy(Y_TEST, Y_PRED)
        print('[Epoch: %d - loss: %.5f | acc: %.5f | prec: %.5f | rec: %.5f | f1: %.5f]' % (
        epoch + 1, running_loss / len(trainloader), acc, precision, recall, fscore))

    print('RNN: Finished Training')
    torch.save(rnn_model.state_dict(), 'saved_models/rnn_model_w_pretrained_cnn')


def classification_accuracy(y_true: list, y_pred: list):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred_class = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_true, y_pred_class)
    precision, recall, fscore, support = score(y_true, y_pred_class, average='macro')

    return acc, precision, recall, fscore


if __name__ == '__main__':
    trainloader_D, testloader_D, data_labels_train, data_labels_test = get_dataloader()

    cnn_model = Anti_spoof_net_CNN.Anti_spoof_net_CNN()
    rnn_model = Anti_spoof_net_RNN.Anti_spoof_net_RNN()

    cnn_criterion = nn.MSELoss()
    rnn_criterion = nn.CrossEntropyLoss()

    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

    if str(sys.argv[1]) == 'cnn':
        train_CNN(cnn_model=cnn_model, optimizer=cnn_optimizer, trainloader=trainloader_D, criterion=cnn_criterion,
                  cnn_device=gpu0, n_epoch=40)

    elif str(sys.argv[1]) == 'rnn':
        cnn_model.load_state_dict(torch.load('saved_models/cnn_model'))
        cnn_model.eval()

        train_RNN(rnn_model=rnn_model, pretrained_cnn=cnn_model, optimizer=rnn_optimizer, trainloader=trainloader_D,
                  spoof_labels=data_labels_train, criterion=rnn_criterion, cnn_device=gpu0, rnn_device=gpu0, n_epoch=40)
