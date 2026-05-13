import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=32, nhead=4):
        super(Transformer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.relu(x)
        x = self.fc(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class GRU(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc1(x[:, -1, :])
        x = self.relu3(x)
        x = self.fc2(x)
        return x


class CNNGRU(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNGRU, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.gru = nn.GRU(128, 64, batch_first=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)

        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc1(x[:, -1, :])
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class CNNTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNTransformer, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc1 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        return x