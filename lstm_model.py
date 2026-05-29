import torch.nn as nn
import torch

"""
Class the holds the initial state of the LSTM and has a function for data to pass
through the layers
"""
class TectonicLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # cell state

        out, _ = self.lstm(x, (h0, c0))  # Run through LSTM - out contains outputs for every time step
        # Take only the last time step's output; out shape is (batch, time_steps, hidden_size)
        out = out[:, -1, :]
        # Pass through linear layer to get prediction
        out = self.linear(out)

        return out