import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN_32s(torch.nn.Module):
    def __init__(self, n_classes):
        super(FCN_32s, self).__init__()
        self.features = torch.nn.Sequential(
            # conv1
            torch.nn.Conv2d(3, 64, 3, padding=100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2, ceil_mode=True),
            # conv2
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2, ceil_mode=True),
            # conv3
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2, ceil_mode=True),
            # conv4
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2, ceil_mode=True),
            # conv5
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # fc1
        self.fc1 = torch.nn.Conv2d(512, 4096, 7) # full-sized kernel
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.drop1 = torch.nn.Dropout2d()

        # fc2
        self.fc2 = torch.nn.Conv2d(4096, 4096, 1) # 1 * 1 卷积调整通道的数目
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.drop2 = torch.nn.Dropout2d()

        # result
        self.result = torch.nn.Conv2d(4096, n_classes, 1)  # 1 * 1 卷积调整通道的数目

        # FCN-32s
        self.upsamples = torch.nn.ConvTranspose2d(n_classes, n_classes, 64, stride=32)



        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(512 * 7 * 7, 512),  # 224x244 image pooled down to 7x7 from features
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Dropout(),
        #     torch.nn.Linear(4096, 4096),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Dropout(),
        #     torch.nn.Linear(4096, n_classes)
        # )

    def forward(self, x):

        data = x
        data = self.features(data)

        print("features 模块后的 shape", data.shape)

        data = self.relu1(self.fc1(data))
        data = self.drop1(data)

        print("self.fc1 后的 shape", data.shape)

        data = self.relu2(self.fc2(data))
        data = self.drop2(data)

        print("self.fc2 后的 shape", data.shape)

        data = self.result(data)

        print("self.result 后的 shape", data.shape)

        result = self.upsamples(data)

        print("self.upsamples 后的 shape", result.shape)

        return result

if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    print("图片的形状", x.shape)
    model = FCN_32s(21)
    model.eval()
    y = model(x)
    print("FCN_32s模型输出的形状", y.shape)
    # y.size()