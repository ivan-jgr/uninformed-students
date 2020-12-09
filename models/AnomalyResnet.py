import torch
import torch.nn as nn
from torchvision import models


class AnomalyResnet(nn.Module):
    def __init__(self):
        super(AnomalyResnet, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        n_filters = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(n_filters, 2)  # Anomaly or not

    def load_checkpoint(self, ckp_path, map_location=False):
        if map_location:
            state_dict = torch.load(ckp_path, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(ckp_path)
        self.backbone.load_state_dict(state_dict)

    def forward(self, x):
        x = self.backbone(x)
        return x
