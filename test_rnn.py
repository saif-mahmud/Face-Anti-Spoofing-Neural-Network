import torch
import torch.nn as nn

from Models import Anti_spoof_net_RNN

if __name__ == '__main__':
    rnn_model = Anti_spoof_net_RNN.Anti_spoof_net_RNN()
    criterion = nn.MSELoss()

    hidden = (torch.zeros(1, 1, 100), torch.zeros(1, 1, 100))

    x = torch.randn(5, 1, 32, 32)
    y, _ = rnn_model(x, hidden)

    norm = torch.linalg.norm(y)
    y = y.view(50)

    print(y)
    print(y.shape)

    print(torch.square(norm))