import torch
import torch.nn as nn


class Sub_Block(nn.Module):
    def __init__(self,in_num,out_num,identity=False):
        super(Sub_Block, self).__init__()
        self.conv0 = nn.Conv3d(in_channels=in_num, out_channels=out_num,
                               kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv3d(in_channels=in_num, out_channels=out_num,
                               kernel_size=1, stride=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(num_features=out_num)
        self.identity = identity

    def forward(self, x):
        if self.identity:
            x1 = self.conv1(x)
        x = self.conv0(x)
        x = self.bn(x)
        if self.identity:
            x = x + x1
        x = self.pool(x)
        x = self.relu(x)

        return x       

