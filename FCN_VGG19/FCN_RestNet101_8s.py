import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import matplotlib.pyplot as plt


class Block(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

class ResBlock(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans):
        super(Bottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out


class Bottleneck1(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans):
        super(Bottleneck1, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)
        self.conv1 = ResBlock(in_chans, out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        identity = self.conv1(identity)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out


class DownBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans):
        super(DownBottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0, stride=2)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, padding=0, stride=2)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        identity = self.conv1(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out


def make_layers(in_channels, layer_list, name="vgg"):
    layers = []
    if name == "vgg":
        for v in layer_list:
            layers += [Block(in_channels, v)]
            in_channels = v
    elif name == "resnet":
        layers += [DownBottleneck(in_channels, layer_list[0])]
        in_channels = layer_list[0]
        for v in layer_list[1:]:
            layers += [Bottleneck(in_channels, v)]
            in_channels = v
    return nn.Sequential(*layers)


class Layer(nn.Module):
    def __init__(self, in_channels, layer_list, net_name):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list, name=net_name)

    def forward(self, x):
        out = self.layer(x)
        return out

class ResNet101(nn.Module):
    '''
    ResNet101 model
    '''
    def __init__(self):
        super(ResNet101, self).__init__()
        self.conv1 = Block(3, 64, 7, 3, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_1 =Bottleneck1(64, 256)
        self.conv2_2 =Bottleneck(256, 256)
        self.conv2_3 =Bottleneck(256, 256)
        self.layer3 = Layer(256, [512, 512, 512, 512], "resnet")
        self.layer4 = Layer(512, [1024]*23, "resnet")
        self.layer5 = Layer(1024, [2048]*3, "resnet")
    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(self.pool1(f1))))
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        return [f3, f4, f5]


class Resnet101_8s(nn.Module):
    def __init__(self, n_class):
        super(Resnet101_8s, self).__init__()
        self.encode = ResNet101()

        self.score_fr = nn.Conv2d(2048, n_class, 1)
        self.trans_p3 = nn.Conv2d(512, n_class, 1)
        self.trans_p4 = nn.Conv2d(1024, n_class, 1)
        self.smooth_conv1 = nn.Conv2d(n_class, n_class, 3, padding=1)
        self.smooth_conv2 = nn.Conv2d(n_class, n_class, 3, padding=1)

        self.up2time = nn.ConvTranspose2d(
            n_class, n_class, 2, stride=2, bias=False)
        self.up4time = nn.ConvTranspose2d(
            n_class, n_class, 2, stride=2, bias=False)
        self.up32time = nn.ConvTranspose2d(
            n_class, n_class, 8, stride=8, bias=False)

    def forward(self, x):
        feature_list = self.encode(x)
        p3, p4, p5 = feature_list

        f7 = self.score_fr(p5)
        up2_feat = self.up2time(f7)

        h = self.trans_p4(p4)
        h = h + up2_feat
        h = self.smooth_conv1(h)

        up4_feat = self.up4time(h)
        h = self.trans_p3(p3)
        h = h + up4_feat
        h = self.smooth_conv2(h)

        final_scores = self.up32time(h)
        return final_scores

if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    fcn_resnet = Resnet101_8s(21)
    fcn_resnet.eval()
    y = fcn_resnet(x)
    print(y.shape)