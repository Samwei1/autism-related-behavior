import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision import models
from models.tcn import TCN


class resnet_lstm(nn.Module):
    def __init__(self, num_classes, hidden_size, fc_size, dropout=0, num_layers=2, bidirectional=False):
        super(resnet_lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size
        self.feature_model = models.resnet152(pretrained=True)
        for i, param in enumerate(self.feature_model.parameters()):
            param.requires_grad = False
        self.feature_model.fc = nn.Linear(self.feature_model.fc.in_features, 500)
        self.fc_pre = nn.Sequential(nn.Linear(500, fc_size), nn.Dropout())
        self.rnn = nn.LSTM(input_size=fc_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=bidirectional)  # LSTM

        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, num_classes)  # logits
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
        self.fc_size = fc_size
        self.softmax = nn.Softmax(dim=-1)

    #  CNN+LSTM
    def forward(self, x, hidden=None, steps=0):
        x = x.squeeze()
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.feature_model(conv_input)
        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_input = self.fc_pre(lstm_input)
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(lstm_input, hidden)
        outputs = self.fc(outputs)
        outputs = torch.mean(outputs, dim=1)
        return self.softmax(outputs)


def my_efficientnet_forward(self, inputs):
    """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
    bs = inputs.size(0)
    # Convolution layers
    x = self.extract_features(inputs)

    # Pooling and final linear layer
    x = self._avg_pooling(x)
    FV = x.view(bs, -1)
    h = self._dropout(FV)
    Logit = self._fc(h)
    return (FV, Logit)


EfficientNet.forward.__code__ = my_efficientnet_forward.__code__


class efficientnet_lstm(nn.Module):

    def __init__(self, num_classes, hidden_size, fc_size, dropout=0, num_layers=2, bidirectional=False):
        super(efficientnet_lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size
        self.pool = nn.AvgPool2d(10)

        self.feature_model = EfficientNet.from_pretrained('efficientnet-b3')
        for i, param in enumerate(self.feature_model.parameters()):
            param.requires_grad = False
        self.fc_pre = nn.Sequential(nn.Linear(1536, fc_size), nn.Dropout())
        self.rnn = nn.LSTM(input_size=fc_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=bidirectional)  # LSTM

        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, num_classes)  # logits
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
        self.fc_size = fc_size
        self.softmax = nn.Softmax(dim=-1)

    #  CNN+LSTM
    def forward(self, x, hidden=None, steps=0):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.feature_model(conv_input)
        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_input = self.fc_pre(lstm_input)
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(lstm_input, hidden)
        outputs = self.fc(outputs)
        outputs = torch.mean(outputs, dim=1)
        return self.softmax(outputs)


class efficientnet_tcn(nn.Module):
    def __init__(self, num_classes, hidden_size, fc_size, dropout=0, k_size=2,level_Size =5, train_cnn = False):
        super(efficientnet_tcn, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size
        self.pool = nn.AvgPool2d(10)

        self.feature_model = EfficientNet.from_pretrained('efficientnet-b3', num_classes= 500)
        for i, param in enumerate(self.feature_model.parameters()):
            param.requires_grad = train_cnn
        self.fc_pre = nn.Sequential(nn.Linear(1536, fc_size), nn.Dropout())
        Num_Chans = [hidden_size] * (level_Size - 1) + [fc_size]
        self.rnn = TCN(fc_size, num_classes, Num_Chans, k_size, dropout, 64)
        self.fc_size = fc_size

    #  CNN+LSTM
    def forward(self, x, hidden=None, steps=0):
        x = x.squeeze(2)
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.feature_model(conv_input)
        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_input = self.fc_pre(lstm_input)
        lstm_input.transpose_(2,1)
        outputs = self.rnn(lstm_input)
        return outputs
