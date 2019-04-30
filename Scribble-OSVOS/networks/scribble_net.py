import torch
import torch.nn as nn
from resnet import *


class ScribbleNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.interaction = InteractionNetwork()
        self.propogation = PropogationNetwork()

    def forward(self, image, prev_mask, prev_time_mask, scribble):
        mask = self.interaction(image, prev_mask, scribble)
        # mask = self.propogation(image, prev_mask, mask)


class InteractionNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)

    def forward(self, image, prev_mask, scribble):
        x = torch.cat((image, prev_mask, scribble), dim=1)
        l1, l2, l3, l4 = self.resnet(x)


class PropogationNetwork(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, image, prev_mask, prev_time_mask):
        pass


if __name__ == '__main__':
    batch = 16
    model = ScribbleNet()
    model.cuda()
    output = model(image=torch.randn(batch, 3, 256, 256).cuda(),
                   prev_mask=torch.randn(batch, 1, 256, 256).cuda(),
                   prev_time_mask=torch.randn(batch, 1, 256, 256).cuda(),
                   scribble=torch.randn(batch, 2, 256, 256).cuda())
    print(output.shape)
