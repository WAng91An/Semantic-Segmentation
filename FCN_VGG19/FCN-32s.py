import math
import torch
import torch.nn as nn
import torch.nn.init as init

# 定义 Block 组件，该组件是一层卷积 + BN + ReLU
class Block(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


# make_layers 给定输入图像 channel 数目，和要经过每层 layer 的通道数据
# 返回多个 layer
def make_layers(in_channels, layer_list):
    layers = []
    for v in layer_list:
        layers += [Block(in_channels, v)]
        in_channels = v
    return nn.Sequential(*layers)

# 定义 Layer 模块，如 VGG-19 中， [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
# [64, 64] 可看作一层，[128, 128] 可以看作一层，
class Layer(nn.Module):
    def __init__(self, in_channels, layer_list):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list)

    def forward(self, x):
        out = self.layer(x)
        return out

# 定义 VGG 模型
class VGG(nn.Module):
    '''
    VGG-19 model
    [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    '''
    def __init__(self):
        super(VGG, self).__init__()
        self.layer1 = Layer(3, [64, 64]) # 输入图像 3 channel，首先经过两个通道数为 64 的 Block
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layer(64, [128, 128]) # 紧接着经过两个通道数为 128 的 Block，两个 Block 可看作一层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layer(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layer(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layer(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        f1 = self.pool1(self.layer1(x))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))
        return [f3, f4, f5] # 返回后三层的特征图


# 上采样模块
class FCNDecode(nn.Module):
    def __init__(self, n, in_channels, out_channels, upsample_ratio):
        super(FCNDecode, self).__init__()
        self.conv1 = Layer(in_channels, [out_channels]*n)
        self.trans_conv1 = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                upsample_ratio,
                stride=upsample_ratio) # 512, 256, 32
    def forward(self, x): # x: [1, 512, 8, 8]
        print("1", self.conv1(x).shape)
        out = self.trans_conv1(self.conv1(x))
        return out

# 建立 FCN-32S 模型
class FCNSeg(nn.Module):
    def __init__(self, n, in_channels, out_channels, upsample_ratio):
        super(FCNSeg, self).__init__()
        self.encode = VGG()
        self.decode = FCNDecode(n, in_channels, out_channels, upsample_ratio)
        self.classifier = nn.Conv2d(out_channels, 10, 3, padding=1)
    def forward(self, x):
        feature_list = self.encode(x)
        print(feature_list[-1].shape)
        out = self.decode(feature_list[-1]) # feature_list[-1]： [1, 512, 8, 8]
        print(out.shape)
        pro = self.classifier(out)
        return pro

if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    model = FCNSeg(4, 512, 256, 32)
    model.eval()
    y = model(x)
    print(y.shape)
    # y.size()