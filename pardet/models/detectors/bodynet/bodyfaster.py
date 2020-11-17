from ..builder import BODYNET
from .base import BaseBody

import torch
from torch.nn import Conv2d, AvgPool2d, Linear, Softmax
from torch.nn import Dropout, BatchNorm2d


@BODYNET.register_module()
class BodyFaster(BaseBody):
    def __init__(self, name, train_cfg, test_cfg):
        super(BodyFaster, self).__init__()

        self.name = name
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.network_define()

        pass

    def network_define(self):
        self.conv1 = Conv2d(3, 8, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = Conv2d(8, 8, (3, 3), stride=(1, 1), padding=(1, 1))
        self.avgpool = AvgPool2d(kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.linear1 = Linear(8, 8)
        self.linear2 = Linear(8, 3)
        self.softmax = Softmax(dim=1)

        self.bn = BatchNorm2d(8)
        self.dropout = Dropout(p=0.5)

        pass

    def extract_feat(self, imgs):
        pass

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.avgpool(net)
        net = self.bn(net)

        net = torch.squeeze(net)
        net = self.linear1(net)
        net = self.dropout(net)
        net = self.linear2(net)
        net = self.dropout(net)

        net = self.softmax(net)

        return net
