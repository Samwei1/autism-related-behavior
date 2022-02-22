import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleStageTCN(nn.Module):
    # Originally written by yabufarha
    # https://github.com/yabufarha/ms-tcn/blob/master/model.py

    def __init__(self, in_channel, n_features, n_classes, n_layers):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2**i, n_features, n_features) for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    # Originally written by yabufarha
    # https://github.com/yabufarha/ms-tcn/blob/master/model.py

    def __init__(self, dilation, in_channel, out_channels):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return x + out


class NormalizedReLU(nn.Module):
    """
    Normalized ReLU Activation prposed in the original TCN paper.
    the values are divided by the max computed per frame
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = F.relu(x)
        x /= x.max(dim=1, keepdim=True)[0] + self.eps

        return x


class EDTCN(nn.Module):
    """
    Encoder Decoder Temporal Convolutional Network
    """

    def __init__(self, in_channels, n_classes, kernel_size=25, mid_channels=[128, 160]):
        """
            Args:
                in_channels: int. the number of the channels of input feature
                n_classes: int. output classes
                kernel_size: int. 25 is proposed in the original paper
                mid_channels: list. the list of the number of the channels of the middle layer.
                            [96 + 32*1, 96 + 32*2] is proposed in the original paper
            Note that this implementation only supports n_layer=2
        """
        super().__init__()

        # encoder
        self.enc1 = nn.Conv1d(
            in_channels, mid_channels[0], kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = NormalizedReLU()

        self.enc2 = nn.Conv1d(
            mid_channels[0], mid_channels[1], kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        self.dropout2 = nn.Dropout(0.3)
        self.relu2 = NormalizedReLU()

        # decoder
        self.dec1 = nn.Conv1d(
            mid_channels[1], mid_channels[1], kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        self.dropout3 = nn.Dropout(0.3)
        self.relu3 = NormalizedReLU()

        self.dec2 = nn.Conv1d(
            mid_channels[1], mid_channels[0], kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        self.dropout4 = nn.Dropout(0.3)
        self.relu4 = NormalizedReLU()

        self.conv_out = nn.Conv1d(mid_channels[0], n_classes, 1, bias=True)

        self.init_weight()

    def forward(self, x):
        # encoder 1
        x1 = self.relu1(self.dropout1(self.enc1(x)))
        t1 = x1.shape[2]
        x1 = F.max_pool1d(x1, 2)

        # encoder 2
        x2 = self.relu2(self.dropout2(self.enc2(x1)))
        t2 = x2.shape[2]
        x2 = F.max_pool1d(x2, 2)

        # decoder 1
        x3 = F.interpolate(x2, size=(t2, ), mode='nearest')
        x3 = self.relu3(self.dropout3(self.dec1(x3)))

        # decoder 2
        x4 = F.interpolate(x3, size=(t1, ), mode='nearest')
        x4 = self.relu4(self.dropout4(self.dec2(x4)))

        out = self.conv_out(x4)

        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class MultiStageTCN(nn.Module):
    """
        Y. Abu Farha and J. Gall.
        MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
        In IEEE Conference on Computer Vision and Pattern Recognition(CVPR), 2019
        parameters used in originl paper:
            n_features: 64
            n_stages: 4
            n_layers: 10
        This MS-TCN class also supports a combination of ED-TCNs and dilated TCNs.
    """

    def __init__(self, in_channel, n_classes, stages,
                 n_features=64, dilated_n_layers=10, kernel_size=15):
        """
            Args:
                in_channel: int. the number of channels of input features
                n_classes: int. the number of output classes of hidden layers
                stages: list. which tcn to be used for each stage.
                        If you want to set this model as the original multi stage tcn,
                        please set['dilated', 'dilated', 'dilated', 'dilated']
                n_features: int. the number of channels of hidden layers
                dilated_n_layers: int. the number of layers of dilated tcn
                kernel_size: int. the kernel size of encoder decoder tcn
        """
        super().__init__()
        if stages[0] == 'dilated':
            self.stage1 = SingleStageTCN(
                in_channel, n_features, n_classes, dilated_n_layers)
        elif stages[0] == 'ed':
            self.stage1 = EDTCN(in_channel, n_classes)
        else:
            print("Invalid values as stages in Mixed Multi Stage TCN")
            sys.exit(1)

        if len(stages) == 1:
            self.stages = None
        else:
            self.stages = []
            for stage in stages[1:]:
                if stage == 'dilated':
                    self.stages.append(SingleStageTCN(
                        n_classes, n_features, n_classes, dilated_n_layers))
                elif stage == 'ed':
                    self.stages.append(
                        EDTCN(n_classes, n_classes, kernel_size=kernel_size))
                else:
                    print("Invalid values as stages in Mixed Multi Stage TCN")
                    sys.exit(1)
            self.stages = nn.ModuleList(self.stages)

    def forward(self, x):
        if self.training:
            # for training
            outputs = []
            out = self.stage1(x)
            outputs.append(out)
            if self.stages is not None:
                for stage in self.stages:
                    out = stage(F.softmax(out, dim=1))
                    outputs.append(out)
            return outputs
        else:
            # for evaluation
            out = self.stage1(x)
            if self.stages is not None:
                for stage in self.stages:
                    out = stage(F.softmax(out, dim=1))
            return out