import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SE_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor onECA the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class SEA_Layer(nn.Module):
    def __init__(self, channel, reduction=16, k_size=3):
        super(SEA_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # super(SEA_Layer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        # y = x * y.expand_as(x)

        # feature descriptor onECA the global spatial information
        # y = self.avg_pool(y)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3, first=False)-> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.eca1 = eca_layer(planes, k_size)
        # self.se1 = SE_Layer(planes, k_size)
        self.sea1 = SEA_Layer(planes, k_size)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        # self.eca = eca_layer(planes * 4, k_size)
        # self.se = SE_Layer(planes * 4, k_size)
        self.sea = SEA_Layer(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride

        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                # Convolutional kernel 1 for ascending and descending dimensions
                # Note that when the jump occurs, there is always a string=2, which is when the output channel dimension is increased every time
                nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(planes * 4))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.se1(out)
        # out = self.eca1(out)
        out = self.sea1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se1(out)
        # out = self.eca(out)
        out = self.sea1(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.se(out)
        # out = self.eca(out)
        out = self.sea(out)
        # out = self.sea1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class Bottleneck(nn.Module):
#     expansion = 4
#     def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         out = self.bn3(out)
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out



class CCA(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(CCA, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        self.conv0 = nn.Conv2d(256, 512, kernel_size=3,stride=2,padding=1)
        self.conv01 = nn.Conv2d(512, 512, kernel_size=3,stride=2,padding=1)
        self.conv02 = nn.Conv2d(256, 512, kernel_size=1,stride=1,padding=0)
        self.eca = eca_layer(self.in_planes, 3)
        self.se = SE_Layer(self.in_planes, 3)
        self.sea = SEA_Layer(self.in_planes, 3)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        # self.conv00 = #  16----8
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y
    def forward(self, x):
        # Bottom-up
        # print(x.shape)
        c1 = F.relu(self.bn1(self.conv1(x)))
        # print(c1.shape)
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        # print(f'c1:{c1.shape}')
        c2 = self.layer1(c1)
        # print(f'c2:{c2.shape}')
        c3 = self.layer2(c2)
        #print(f'c3:{c3.shape}')
        c4 = self.layer3(c3)
        # print(f'c4:{c4.shape}')
        c5 = self.layer4(c4)
        # print(f'c5:{c5.shape}')
        # Top-down
        p5 = self.toplayer(c5)
        # p5 = self.eca(p5)
        # print(f'p5:{p5.shape}')
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        # p4 = self.eca(p4)
        # print(f'latlayer1(c4):{self.latlayer1(c4).shape}, p4:{p4.shape}')
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # p3 = self.eca(p3)
        #print(f'latlayer1(c3):{self.latlayer2(c3).shape}, p3:{p3.shape}')
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # p2 = self.eca(p2)
        #print(f'latlayer1(c2):{self.latlayer3(c2).shape}, p2:{p2.shape}')
        # Smooth
        p4 = self.smooth1(p4)
        # print(p4.shape)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        # out = torch.cat([p2, p3], 1)
        # p2 = p2.view(p2.size(0), -1)
        #p2 Reduce dimensionality to 4x4
        p2 = self.conv0(p2)
        # print(p2.shape)
        p2 = self.conv01(p2)
        p2 = self.conv01(p2)
        p2 = self.relu(p2)

        #p3 Reduce dimensionality to 4x4
        p3 = self.conv0(p3)
        p3 = self.conv01(p3)
        p3 = self.relu(p3)
        #p4 Reduce dimensionality to 4x4
        p4 = self.conv0(p4)
        p4 = self.relu(p4)
        #p5
        p5 = self.conv02(p5)
        p5 = self.relu(p5)

        x = torch.cat([p2, p3, p4, p5], 1)
        # x = self.se(x)
        # x = self.eca(x)
        # print(x.shape)
        x = self.sea(x)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print(x.shape)

        return x
def CCA101():
    return CCA(Bottleneck, [2,4,23,3])
    # return CCA(Bottleneck, [3, 4, 6, 3])
def test():
    net = CCA101()
    fms = net(Variable(torch.randn(128, 3, 128, 128)))
    for fm in fms:
        print(fm.size())

# CCANet_test()