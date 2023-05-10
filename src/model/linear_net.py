import torch
import torch.nn as nn


class LinearNet(nn.Module):
    """Linear classifier"""
    def __init__(self, n_input, dropout_rate, n_out, activation=False):
        super(LinearNet, self).__init__()
        n_hidden = [1024, 512, 256, 128]
        self.dropout_rate = dropout_rate
        self.n_out = n_out
        self.linear = nn.Sequential(
            nn.Linear(n_input, n_hidden[0], bias=False),
            nn.BatchNorm1d(n_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(n_hidden[0], n_hidden[1], bias=False),
            nn.BatchNorm1d(n_hidden[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(n_hidden[1], n_hidden[2], bias=False),
            nn.BatchNorm1d(n_hidden[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(n_hidden[2], n_hidden[3], bias=False),
            nn.BatchNorm1d(n_hidden[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )
        if activation:
            self.fc = nn.Sequential(nn.Linear(n_hidden[-1], n_out),
                                    nn.Tanh())
        else:
            self.fc = nn.Linear(n_hidden[-1], n_out)

        self.linear.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, features):
        out = self.linear(features)
        out = self.fc(out)
        return out
    