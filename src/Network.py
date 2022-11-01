import torch.nn as nn
from src import Sub_block as sb


class My_Network(nn.Module):
    def __init__(self):
        super(My_Network,self).__init__()
        self.block1 = sb.Sub_Block(1, 32)
        self.block2 = sb.Sub_Block(32, 64)
        self.block3 = sb.Sub_Block(64, 128)
        self.block4 = sb.Sub_Block(128, 256)
        self.block5 = sb.Sub_Block(256, 256)
        self.conv1 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(in_channels=128, out_channels=73, kernel_size=1, stride=1)
        #self.fc = nn.Linear(138,53)
        self.bn = nn.BatchNorm3d(num_features=128)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout3d(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        size = x.size(0)
        # print(x.shape)
        x = self.block1.forward(x)
        # print(x.shape)
        x = self.block2.forward(x)
        # print(x.shape)
        x = self.block3.forward(x)
        # print(x.shape)
        x = self.block4.forward(x)
        # print(x.shape)
        x = self.block5.forward(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        #x = self.pool(x)
        x = nn.functional.adaptive_avg_pool3d(x,(1,1,1))
        x = self.dropout(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(size, -1)
        # print(x.shape)
        return x

