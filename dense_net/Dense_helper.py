import torch
import torch.nn as nn
from cfg import *

class DenseNet(nn.Module):
    def __init__(self, init=GROWTH, dense_layer_lst=DENSE_BNUM):
        super(DenseNet, self).__init__()

        self.dblock_num = len(dense_layer_lst)
        self.init = init
        self.dims = self._calulate_dim(dense_layer_lst)
        self.dense = nn.ModuleList([])
        self.subs = nn.ModuleList([])

        for layer_id in range(self.dblock_num):
            unit = DenBlock_B(cin=self.dims[layer_id], num_layers=DENSE_BNUM[layer_id], is_downsampling=True)
            # A Gate to control how much info flow from here into the element-wise summation
            self.subs.append(nn.Conv2d(self.dims[layer_id + 1], self.dims[layer_id + 1], (1, 1)))
            self.dense.append(unit)

    def _calulate_dim(self, lst):
        ini = self.init
        dims = []
        for ele in lst:
            dims.append(ini)
            ini = (ini + ele * GROWTH) // 2
        last = (dims[-1] + lst[-1] * GROWTH) // 2
        dims.append(last)

        return dims

    def forward(self, x):
        subouts = []
        for i in range(len(self.dense)):
            x = self.dense[i](x)
            x = self.subs[i](x)
            subouts.append(x)
        sub1, sub2, sub3, sub4 = subouts

        return sub1, sub2, sub3, sub4


class Conv(nn.Module):
    '''
    the input layer before dense block
    eg.
        input shape: (1, 3, 128, 128)
        output shape: (1, 8, 64, 64)
    '''

    def __init__(self, cin=3, cout=GROWTH, is_large_feature=False):
        super(Conv, self).__init__()
        self.cin = cin
        self.cout = cout
        if is_large_feature:
            self.filter_size, self.pad = (3, 3), (1, 1)
        else:
            self.filter_size, self.pad = (5, 5), (2, 2)

        self.layers = nn.Sequential(nn.Conv2d(self.cin, self.cout, kernel_size=self.filter_size, padding=self.pad),
                                    nn.BatchNorm2d(self.cout),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        out = self.layers(x)
        return out


class DenBlock_B(nn.Module):
    '''
    dense block with bottle neck structure.
    dose downsampling at the end of each block except the last block

    eg.
        input shape: (1, 8, 64, 64)
        output shape: (1, 8, 32, 32)
    '''

    def __init__(self, cin, K=GROWTH, num_layers=5, is_downsampling=True):
        super(DenBlock_B, self).__init__()
        self.cin = cin
        self.din = self.cin + num_layers * K
        self.dense_layers = nn.ModuleList([])
        self.is_downsampling = is_downsampling
        count = 0

        while count < num_layers:
            unit = nn.Sequential(
                nn.BatchNorm2d(self.cin + count * K),
                nn.ReLU(),
                nn.Conv2d(self.cin + count * K, 4 * K, kernel_size=(1, 1)),
                nn.BatchNorm2d(4 * K),
                nn.ReLU(),
                nn.Conv2d(4 * K, K, kernel_size=(3, 3), padding=(1, 1))
            )

            self.dense_layers.append(unit)
            count += 1

        self.down_sampling = nn.Sequential(
            nn.BatchNorm2d(self.din),
            # compressing
            nn.Conv2d(self.din, self.din // 2, kernel_size=(1, 1)),
            # downsampling
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        for seq in self.dense_layers:
            res = x
            x = seq(x)
            x = torch.cat((x, res), dim=1)

        if self.is_downsampling:
            x = self.down_sampling(x)

        return x

class Output(nn.Module):

    def __init__(self, cin):
        super(Output, self).__init__()
        self.layer1 = nn.Conv2d(cin, 8*GROWTH, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(8*GROWTH)
        self.layer2 = nn.Conv2d(8*GROWTH, 4*GROWTH, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(4*GROWTH)
        self.layer3 = nn.Conv2d(4*GROWTH, 2*GROWTH, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(2 * GROWTH)
        self.layer4 = nn.Conv2d(2 * GROWTH, GROWTH, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(GROWTH)
        self.layer5 = nn.Conv2d(GROWTH, CLAS, kernel_size=(3, 3), padding=(1, 1))

        self.adaptive = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.layer5(x)
        x = self.adaptive(x)

        return x


class Fusiontest(nn.Module):

    def __init__(self):
        super(Fusiontest, self).__init__()
        self.conv1 = nn.Conv2d(168, 372, kernel_size=(1, 1), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(372)

        self.conv2 = nn.Conv2d(372, 762, kernel_size=(1, 1), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(762)

        self.conv3 = nn.Conv2d(762, 762, kernel_size=(1, 1), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(762)

    def forward(self, x):

        out = self.conv1(x[0])
        out += x[1]
        out = self.bn1(out)

        out = self.conv2(out)
        out += x[2]
        out = self.bn2(out)

        out = self.conv3(out)
        out += x[3]
        # out = self.bn3(out)

        return out



