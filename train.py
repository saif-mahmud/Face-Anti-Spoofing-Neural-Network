import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Models import Anti_Spoof_net

warnings.filterwarnings("ignore")

print('GPU count:', torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)


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


def train_CNN(model, optimizer, trainloader, criterion, n_epoch=10):

    for epoch in range(n_epoch):

        print('[CNN Epoch: %2d]' % (epoch + 1))

        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            images, labels_D = data
            images, labels_D = images.to(device), labels_D.to(device)

            optimizer.zero_grad()

            outputs_D, _ = model(images)

            # handle NaN:
            if torch.norm((outputs_D != outputs_D).float()) == 0:
                loss = criterion(outputs_D, labels_D)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if i % 100 == 0:
                print('[CNN: Epoch: %d, Batch: %5d] loss: %.3f' % (epoch + 1, i, running_loss / len(trainloader)))

    print('CNN: Finished Training')
    torch.save(model, 'saved_models/cnn_model')


def train_RNN(net, optimizer, trainloader, labels, criterion, n_epoch=10):
    total = 0

    for epoch in range(n_epoch):
        # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            print(i)
            # Donnees pre-crees:
            images, labels_D = data
            images, labels_D = images.to(device), labels_D.to(device)
            # training step

            optimizer.zero_grad()
            _, outputs_F = net(images)

            # handle NaN:
            if torch.norm((outputs_F != outputs_F).float()) == 0:
                # if i % 50 == 0 or i % 50 == 1:
                #     imshow_np(np.transpose(images[0,:,:,:].cpu().numpy(), (1,2,0)))
                #     print('F:')
                #     print(outputs_F)

                if labels[i * 5] == 0:  # all the images in the batch come from the same video
                    label = torch.zeros((5, 1, 2), dtype=torch.float32)
                else:
                    label = torch.ones((5, 1, 2), dtype=torch.float32)

                label = label.to(device)

                loss = criterion(outputs_F, label)
                loss.backward()
                optimizer.step()

                # compute statistics
                total += labels_D.size(0)
                running_loss += loss.item()

                print('[RNN: Epoch : %d, Batch: %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / total))

        print('Epoch finished')

    print('Finished Training')


def train_All(net, optimizer, trainloader, labels, criterion, n_epoch=10):
    for i in range(n_epoch):

        train_CNN(model=net, optimizer=optimizer, trainloader=trainloader, criterion=criterion, n_epoch=10)

        # train_RNN(net=net, optimizer=optimizer, trainloader=trainloader, labels=labels, criterion=criterion, n_epoch=1)



if __name__ == '__main__':
    trainloader_D, testloader_D, data_labels_train, data_labels_test = get_dataloader()

    mon_model = Anti_Spoof_net.Anti_spoof_net()
    mon_model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mon_model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

    train_All(net=mon_model, optimizer=optimizer, trainloader=trainloader_D, labels=data_labels_train,
              criterion=criterion, n_epoch=1)
