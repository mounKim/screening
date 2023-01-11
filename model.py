import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, args):
        self.args = args
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=128, out_channels=100, kernel_size=(3,), dilation=(4,))
        self.fc1 = nn.Linear(9200, 100)
        self.fc2 = nn.Linear(100, 2)
        self.our_embedding = nn.Embedding(10000, 100)
        self.our_embedding.weight.requires_grad = False

    def forward(self, x):
        if not self.args.use_embedding:
            x = self.our_embedding(x.long())
        x = F.relu(self.conv(x))
        y = []
        for i in range(100):
            y.append(x[:, i, :])
        x = torch.cat(y, dim=1)
        x = self.fc2(self.fc1(x))
        return x


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.fc = nn.Linear(100, 2)
        self.our_embedding = nn.Embedding(10000, 100)
        self.our_embedding.weight.requires_grad = False

    def forward(self, x):
        if not self.args.use_embedding:
            x = self.our_embedding(x.long())
        x = torch.tanh(self.lstm(x)[1][0]).squeeze(0)
        x = self.fc(x)
        return x
