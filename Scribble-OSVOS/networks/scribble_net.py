import torch
import torch.nn as nn
import torchvision.models as models

class ScribbleNet(nn.Module):

    def __init__(self):
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)

    def forward(self, x):
        x = resnet18(x)
        return x
