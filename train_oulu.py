import os
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

BATCH_SIZE = 10
LAMBDA = 0.015


def imshow_np(i, img, mode: str):
    height, width, depth = img.shape
    img = np.divide(img, 255)

    if depth == 1:
        img = img[:, :, 0]

    fname = 'sample_img/' + str(i) + mode + '.png'
    plt.imsave(fname, img)


def thresh_plot(score_data):
    fig, ax = plt.subplots()
    colors = {0: 'green', 1: 'red'}
    ax.scatter(score_data[:, 0], score_data[:, 1], c=np.vectorize(colors.get)(score_data[:, 2]))

    plt.grid(True)
    plt.savefig('thresh_plt_oulu.png')


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


def get_rppg(rppg_dir, subdir):
    rppg = list()

    for ppg_file in sorted(os.listdir(os.path.join(rppg_dir, subdir))):
        ppg = np.load(os.path.join(rppg_dir, subdir, ppg_file))
        rppg.append(ppg)

    return np.array(rppg)


def get_oulu_data(split: str, mode='cnn'):
    frames_dir = 'oulu_processed/frames'
    depth_dir = 'oulu_processed/depth_map'
    rppg_dir = 'oulu_processed/rppg'
    labels_dir = 'oulu_processed/labels'

    subdirs = {'train': 'Train_files', 'dev': 'Dev_files', 'test': 'Test_files'}

    if mode == 'cnn':
        frames = np.load(os.path.join(frames_dir, subdirs[split], 'frames.npy'))
        labels = np.load(os.path.join(labels_dir, subdirs[split], 'labels.npy'))

    elif mode == 'rnn':
        frames = np.load(os.path.join(frames_dir, subdirs[split], 'rppg_frames.npy'))
        labels = np.load(os.path.join(labels_dir, subdirs[split], 'rppg_labels.npy'))

    print(frames.shape)
    print(labels.shape)

    dmap = np.load(os.path.join(depth_dir, subdirs[split], 'depth_maps.npy'))
    rppg = get_rppg(rppg_dir, subdirs[split])

    print(dmap.shape)
    print(rppg.shape)

    if mode == 'cnn':
        dataset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(frames, (0, 3, 1, 2)), dtype=torch.float32),
                                                 torch.tensor(dmap, dtype=torch.float32))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        rppg_data = torch.tensor(rppg, dtype=torch.float32)

        if split is 'train':
            return data_loader
        else:
            return data_loader, labels, rppg_data

    elif mode == 'rnn':
        img_dataset = torch.utils.data.TensorDataset(
            torch.tensor(np.transpose(frames, (0, 3, 1, 2)), dtype=torch.float32))
        img_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=10, shuffle=False, drop_last=True)
        rppg_data = torch.tensor(rppg, dtype=torch.float32)

        return img_dataloader, rppg_data


def train_CNN(cnn_model, optimizer, trainloader, criterion, cnn_device, n_epoch):
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
                    imshow_np(i, np.transpose(images[0, :, :, :].cpu().numpy(), (1, 2, 0)), mode='_raw')
                    imshow_np(i, np.transpose(outputs_D[0, :, :, :].cpu().detach().numpy(), (1, 2, 0)),
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
        }, 'checkpoints/oulu/cnn.pt')

    print('CNN: Finished Training')
    torch.save(cnn_model.state_dict(), 'saved_models/oulu/cnn_model')


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
            # print(outputs_F.shape)
            outputs_F = outputs_F.view(50)

            # handle NaN:
            if torch.norm((outputs_F != outputs_F).float()) == 0:
                loss = criterion(outputs_F, rppg_label[i])
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        print('[Epoch: %d - rppg MSE loss: %.5f' % (epoch + 1, running_loss / len(trainloader)))

        # model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': cnn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': (running_loss / len(trainloader)),
        }, 'checkpoints/oulu/rnn.pt')

    print('RNN: Finished Training')
    torch.save(rnn_model.state_dict(), 'saved_models/oulu/rnn_model_rppg')


def classification_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = score(y_true, y_pred, average='macro')

    return acc, precision, recall, fscore


def predict(testloader, spoof_labels, rppg_label, cnn_model, rnn_model, device):
    threshold = 0.1

    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    rnn_model = rnn_model.to(device)
    rnn_model.eval()

    one = torch.ones(BATCH_SIZE, 1, 32, 32).to(device)
    zero = torch.zeros(BATCH_SIZE, 1, 32, 32).to(device)

    hidden = (torch.zeros(1, 1, 100, device=device), torch.zeros(1, 1, 100, device=device))

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

            norm_D = torch.linalg.norm(D[-1, :, :, :])
            norm_F = torch.linalg.norm(outputs_F)

            score = torch.square(norm_F) + (LAMBDA * torch.square(norm_D))
            with open('./training_log/val_scores.csv', 'a') as fd:
                csv_row = '\n' + \
                          str(i) + ', ' + \
                          str(label) + ', ' + \
                          str(norm_D.cpu().detach().numpy()) + ', ' + \
                          str(norm_F.cpu().detach().numpy()) + ', ' + \
                          str(score.cpu().detach().numpy())
                fd.write(csv_row)

            score_data.append(list((i, float(score.cpu().detach().numpy()), float(label))))

            fig, ax = plt.subplots()
            ax.plot(outputs_F.cpu().detach().numpy())
            ax.plot(rppg_label[i])
            plt.grid(True)
            plt.ylabel('rppg_signal')
            plt.title(str('Label : ' + CLASS_NAMES[int(label)] + ' | rPPG Norm: ' + str(norm_F.cpu().detach().numpy())))
            rppg_path = 'rppg_vis_oulu/' + str('val_batch_' + str(i) + '_' + CLASS_NAMES[int(label)])
            plt.savefig(rppg_path)

    print(score_data)
    np.save('thresh_test.npy', score_data)
    thresh_plot(np.array(score_data))


if __name__ == '__main__':
    cnn_model = Anti_spoof_net_CNN.Anti_spoof_net_CNN()
    rnn_model = Anti_spoof_net_RNN.Anti_spoof_net_RNN()

    criterion = nn.MSELoss()

    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

    if str(sys.argv[1]) == 'cnn':
        trainloader_D = get_oulu_data(split='train', mode='cnn')
        train_CNN(cnn_model=cnn_model, optimizer=cnn_optimizer, trainloader=trainloader_D, criterion=criterion,
                  cnn_device=gpu0, n_epoch=1)

    elif str(sys.argv[1]) == 'rnn':
        img_dataloader, rppg_data = get_oulu_data(split='train', mode='rnn')
        cnn_model.load_state_dict(torch.load('saved_models/oulu/cnn_model'))
        cnn_model.eval()

        train_RNN(rnn_model=rnn_model, pretrained_cnn=cnn_model, optimizer=rnn_optimizer, trainloader=img_dataloader,
                  rppg_label=rppg_data, criterion=criterion, cnn_device=gpu0, rnn_device=gpu0, n_epoch=75)

    elif str(sys.argv[1]) == 'pred':
        cnn_model.load_state_dict(torch.load('saved_models/oulu/cnn_model'))
        rnn_model.load_state_dict(torch.load('saved_models/oulu/rnn_model_rppg'))

        testloader_D, data_labels_test, rppg_label = get_oulu_data(split='dev')

        predict(testloader=testloader_D, spoof_labels=data_labels_test, rppg_label=rppg_label, cnn_model=cnn_model, rnn_model=rnn_model,
                device=gpu0)
