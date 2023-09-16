import torch.nn as nn

class  BasicBlock(nn.Module):
    expansion = 1 # Expansion coefficient
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # 短连接方式
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        x = nn.ReLU(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + self.shortcut(x)
        out = nn.ReLU(x)

        return out

# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, self.expansion *
#                                out_channels, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != self.expansion*out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, self.expansion*out_channels,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*out_channels)
#             )
#
#     def forward(self, x):
#         out = nn.ReLU(self.bn1(self.conv1(x)))
#         out = nn.ReLU(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = nn.ReLU(out)
#         return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3),
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.__make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.__make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.__make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.__make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_blocks)

    def __make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels*block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.ReLU(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.avg_pool2d(x, 4)
        out = self.fc(x)

        return out

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
