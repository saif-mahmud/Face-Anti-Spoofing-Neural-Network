import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Anti_spoof_net_RNN(nn.Module):
    def __init__(self):
        super(Anti_spoof_net_RNN, self).__init__()

        self.hidden_dim = 100
        self.input_dim = 32 * 32
        self.num_layers = 1

        self.LSTM = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers)
        self.fc = nn.Linear(self.hidden_dim, 50)

    def forward(self, F, hidden):
        batch_size, _, _, _ = F.size()
        F = F.view(batch_size, 1, -1)

        lstm_out, hidden = self.LSTM(F, hidden)  # lstm_out: [batch_size, 1, 100]
        R = self.fc(lstm_out)  # R: [batch_size, 1, 50]
        R = torch.fft(R, signal_ndim=1, normalized=True)

        return R, hidden
