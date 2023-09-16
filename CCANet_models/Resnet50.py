import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0],first=False) -> None:
        super(Bottleneck,self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # Replace in place to save memory overhead
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding[1],bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # Replace in place to save memory overhead
            nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=stride[2], padding=padding[2],bias=False),
            nn.BatchNorm2d(out_channels*4)
        )

        # shortcut part
        # Due to inconsistent dimensions, there are different situations
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                # Convolutional kernel 1 for ascending and descending dimensions
                # Note that when the jump occurs, there is always a string=2, which is when the output channel dimension is increased every time
                nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels*4)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# In networks using BN, the output of the convolutional layer is not biased
class ResNet(nn.Module):
    def __init__(self, Bottleneck, num_classes=1) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Adding attention mechanism to the first layer of the network


        # The first layer serves as a separate layer because there is no residual fast
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # conv2
        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)

        # conv3
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)

        # conv4
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)

        # conv5
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)

        # Adding attention mechanism to the last layer of the convolutional layer of the network



        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        # Used to determine if it is the first layer of each block layer
        flag = True
        for i in range(0, len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet50():
    return ResNet(Bottleneck)

# res50 = ResNet50(Bottleneck)
# print(res50)

