import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN_16s(torch.nn.Module):
    def __init__(self, n_classes):
        super(FCN_16s, self).__init__()

        # conv1
        self.conv1_1 = torch.nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = torch.nn.ReLU(inplace=True)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv2
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = torch.nn.ReLU(inplace=True)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv3
        self.conv3_1 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = torch.nn.ReLU(inplace=True)
        self.conv3_2 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = torch.nn.ReLU(inplace=True)
        self.conv3_3 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = torch.nn.ReLU(inplace=True)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv4
        self.conv4_1 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = torch.nn.ReLU(inplace=True)
        self.conv4_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = torch.nn.ReLU(inplace=True)
        self.conv4_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = torch.nn.ReLU(inplace=True)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv5
        self.conv5_1 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = torch.nn.ReLU(inplace=True)
        self.conv5_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = torch.nn.ReLU(inplace=True)
        self.conv5_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = torch.nn.ReLU(inplace=True)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # fc6 - > conv6
        self.conv6 = torch.nn.Conv2d(512, 4096, 7) # full-sized kernel
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.drop6 = torch.nn.Dropout2d()

        # fc7 - > conv7
        self.conv7 = torch.nn.Conv2d(4096, 4096, 1) # 1 * 1 卷积调整通道的数目
        self.relu7 = torch.nn.ReLU(inplace=True)
        self.drop7 = torch.nn.Dropout2d()

        # result
        self.result = torch.nn.Conv2d(4096, n_classes, 1)  # 1 * 1 卷积调整通道的数目

        # FCN-16s
        self.upsamples_2x = torch.nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2) # x2
        self.upsamples_16x = torch.nn.ConvTranspose2d(n_classes, n_classes, 32, stride=16) # x16

        
    def forward(self, x):
        data = x
        data = self.relu1_1(self.conv1_1(data))
        data = self.relu1_2(self.conv1_2(data))
        data = self.pool1(data)

        data = self.relu2_1(self.conv2_1(data))
        data = self.relu2_2(self.conv2_2(data))
        data = self.pool2(data)

        data = self.relu3_1(self.conv3_1(data))
        data = self.relu3_2(self.conv3_2(data))
        data = self.relu3_3(self.conv3_3(data))
        data = self.pool3(data)

        data = self.relu4_1(self.conv4_1(data))
        data = self.relu4_2(self.conv4_2(data))
        data = self.relu4_3(self.conv4_3(data))
        data = self.pool4(data)

        pool4 = data

        data = self.relu5_1(self.conv5_1(data))
        data = self.relu5_2(self.conv5_2(data))
        data = self.relu5_3(self.conv5_3(data))
        data = self.pool5(data)

        data = self.relu6(self.conv6(data))
        data = self.drop6(data)

        data = self.relu7(self.conv7(data))
        data = self.drop7(data)

        data = self.result(data)

        upsamples_2x_result = self.upsamples_2x(data) # 2x

        pool4_upsamples2_sum = pool4 + upsamples_2x_result # 2x result + pool4 result

        result = self.upsamples_16x(pool4_upsamples2_sum) # sum result -> 16x

        return result

if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    print("图片的形状", x.shape)
    model = FCN_16s(21)
    model.eval()
    y = model(x)
    print("FCN_16s模型输出的形状", y.shape)
    # y.size()