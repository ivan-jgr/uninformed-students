import torch.nn as nn


class PatchAnomalyNet(nn.Module):
    def __init__(self):
        super(PatchAnomalyNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, 5, 1)
        self.conv2 = nn.Conv2d(128, 128, 5, 1)
        self.conv3 = nn.Conv2d(128, 256, 5, 1)
        self.conv4 = nn.Conv2d(256, 256, 4, 1)
        self.conv5 = nn.Conv2d(256, 128, 1, 1)
        self.dropout2d = nn.Dropout2d(0.2)
        self.decode = nn.Linear(128, 512)
        self.dropout = nn.Dropout(0.2)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.l_relu = nn.LeakyReLU(5e-3)

    def forward(self, x):
        x = self.l_relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.l_relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.l_relu(self.conv3(x))
        x = self.max_pool(x)
        x = self.l_relu(self.conv4(x))
        x = self.l_relu(self.conv5(x))
        x = self.dropout2d(x)
        x = x.view(-1, 128)
        x = self.l_relu(self.decode(x))
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    import torch
    from torchvision import models

    resnet = models.resnet18(pretrained=False)
    resnet = nn.Sequential(*list(resnet.children())[:-1])

    net = PatchAnomalyNet()
    x = torch.rand((5, 3, 65, 65))
    y_resnet = resnet(x)
    y_net = net(x)

    print("Resnet Size", torch.squeeze(y_resnet).size())
    print("Net Size", y_net.size())