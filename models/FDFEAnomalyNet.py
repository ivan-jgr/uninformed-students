import torch
import torch.nn as nn
from .FDFE import *


class FDEAnomalyNet(nn.Module):
    def __init__(self, base_net, pH, pW, imH, imW, sL1 ,sL2, sL3):
        super(FDEAnomalyNet, self).__init__()
        self.imH = imH
        self.imW = imW

        self.multiPoolPrepare = multiPoolPrepare(pH, pW)
        self.conv1 = base_net.conv1
        self.multiMaxPooling1 = multiMaxPooling(sL1, sL1, sL1, sL1)
        self.conv2 = base_net.conv2
        self.multiMaxPooling2 = multiMaxPooling(sL2, sL2, sL2, sL2)
        self.conv3 = base_net.conv3
        self.multiMaxPooling3 = multiMaxPooling(sL3, sL3, sL3, sL3)
        self.conv4 = base_net.conv4
        self.conv5 = base_net.conv5

        self.outChans = self.conv5.out_channels
        self.unwrapPrepare = unwrapPrepare()
        self.unwrapPool3 = unwrapPool(self.outChans, imH / (sL1 * sL2 * sL3), imW / (sL1 * sL2 * sL3), sL3, sL3)
        self.unwrapPool2 = unwrapPool(self.outChans, imH / (sL1 * sL2), imW / (sL1 * sL2), sL2, sL2)
        self.unwrapPool1 = unwrapPool(self.outChans, imH / sL1, imW / sL1, sL1, sL1)

        self.decode = base_net.decode
        self.decodeChans = self.decode.out_features
        self.l_relu = base_net.l_relu

    def forward(self, x):
        x = self.multiPoolPrepare(x)

        x = self.l_relu(self.conv1(x))
        x = self.multiMaxPooling1(x)

        x = self.l_relu(self.conv2(x))
        x = self.multiMaxPooling2(x)

        x = self.l_relu(self.conv3(x))
        x = self.multiMaxPooling3(x)

        x = self.l_relu(self.conv4(x))
        x = self.l_relu(self.conv5(x))

        x = self.unwrapPrepare(x)
        x = self.unwrapPool3(x)
        x = self.unwrapPool2(x)
        x = self.unwrapPool1(x)

        y = x.view(self.outChans, self.imH, self.imW, -1)
        y = y.permute(3, 1, 2, 0)
        y = self.l_relu(self.decode(y))

        return y


if __name__ == '__main__':
    from PatchNet import PatchAnomalyNet
    import numpy as np

    imH = 224
    imW = 224

    sL1 = 2
    sL2 = 2
    sL3 = 2

    pW = 65
    pH = 65

    testImage = torch.randn(1, 3, imH, imW)

    imW = int(np.ceil(imW / (sL1 * sL2 * sL3)) * sL1 * sL2 * sL3)
    imH = int(np.ceil(imH / (sL1 * sL2 * sL3)) * sL1 * sL2 * sL3)

    base_net = PatchAnomalyNet()
    base_net.eval()

    fdfe_net = FDEAnomalyNet(base_net=base_net, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
    fdfe_net.eval()

    y1 = fdfe_net(testImage)
    print("y1 shape", y1.size())

    pad_top = np.ceil(pH / 2).astype(int)
    pad_bottom = np.floor(pH / 2).astype(int)
    pad_left = np.ceil(pW / 2).astype(int)
    pad_right = np.floor(pW / 2).astype(int)

    cY = np.random.randint(pH - pad_top, imH - pad_bottom)
    cX = np.random.randint(pW - pad_left, imW - pad_right)

    patch = testImage[:, :, cY-pad_bottom:cY+pad_top, cX-pad_right:cX+pad_left]
    print(f'patch center: ({cY}, {cX}), patch size: {patch.size()}')

    y2 = base_net(testImage[:, :, cY - pad_bottom:cY + pad_top, cX - pad_right:cX + pad_left])
    print(y2.size())
    err = torch.mean((y1[:, cY, cX, :] - y2) ** 2).item()
    print(f'error on pixel ({cY}, {cX}): {err}')

    mean_val_pixel_y1 = torch.mean((y1[:, cY, cX, :] ** 2)).item()
    print(mean_val_pixel_y1)

    mean_val_pixel_y2 = torch.mean(y2 ** 2).item()
    print(mean_val_pixel_y2)

    rel_err = 2 * err / (mean_val_pixel_y1 + mean_val_pixel_y2)
    print(f'Relative error: {rel_err}')
