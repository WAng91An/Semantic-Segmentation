import math
import torch
import torch.nn as nn
import numpy as np

# 定义 Block 组件，该组件是一层卷积 + BN + ReLU
class Block(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


# make_layers 给定输入图像 channel 数目，和要经过每层 layer 的通道数据
# 返回多个 layer， layer_list[64, 64]
def make_layers(in_channels, layer_list):
    layers = []
    for v in layer_list:
        layers += [Block(in_channels, v)]
        in_channels = v
    return nn.Sequential(*layers)


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
    VGG model
    '''
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = Layer(64, [64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layer(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layer(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layer(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layer(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        f0 = self.relu1(self.bn1(self.conv1(x)))
        f1 = self.pool1(self.layer1(f0))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))
        return [f3, f4, f5]

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    center = kernel_size / 2
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

class VGG_19_8s(nn.Module):
    def __init__(self, n_class):
        super(VGG_19_8s, self).__init__()
        self.encode = VGG()

        self.fc6 = nn.Conv2d(512, 4096, 7)  # padding=0
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)

        self.trans_p3 = nn.Conv2d(256, n_class, 1)
        self.trans_p4 = nn.Conv2d(512, n_class, 1)

        self.upsamples2x = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        self.upsamples4x = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        self.upsamples32x = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = bilinear_kernel(n_class, n_class, m.kernel_size[0])

    def forward(self, x):
        feature_list = self.encode(x)
        p3, p4, p5 = feature_list

        print("p3 shape:", p3.shape)
        print("p4 shape:", p4.shape)
        print("p5 shape:", p5.shape)

        f6 = self.drop6(self.relu6(self.fc6(p5)))
        print("f6 shape", f6.shape)

        f7 = self.score_fr(self.drop7(self.relu7(self.fc7(f6))))
        print("f7 shape", f7.shape)

        up2_feat = self.upsamples2x(f7)
        print("up2_feat shape", up2_feat.shape)
        h = self.trans_p4(p4)
        print("p4 shape", p4.shape)
        print("self.trans_p4(p4) shape", h.shape)
        h = h[:, :, 5:5 + up2_feat.size()[2], 5:5 + up2_feat.size()[3]]

        print("crop1 shape", h.shape)

        h = h + up2_feat

        print("summation_1 shape", h.shape)

        up4_feat = self.upsamples4x(h)

        print("up4_feat shape", up4_feat.shape)

        h = self.trans_p3(p3)
        print("self.trans_p3(p3) shape", h.shape)

        h = h[:, :, 9:9 + up4_feat.size()[2], 9:9 + up4_feat.size()[3]]
        print("crop2 shape", h.shape)
        h = h + up4_feat
        print("summation_2 shape", h.shape)

        h = self.upsamples32x(h)
        final_scores = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return final_scores

if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    model = VGG_19_8s(21)
    model.eval()
    y = model(x)
    print(y.shape)