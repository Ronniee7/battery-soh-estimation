
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=3, hidden_size=32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.regressor(output[:, -1, :])
