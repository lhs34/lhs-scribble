import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Encoder
        self.resnet = resnet18(pretrained=True, input_layers=6)

        # Aggregation block
        self.feature_aggregation = AggregateBlock()

        # Decoder
        self.decoder1 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder3 = DecoderBlock(128, 64)
        self.trans_conv = nn.ConvTranspose2d(64, 1, kernel_size=3)

    def forward(self, image, prev_mask, scribble, prev_agg):
        # Encoder
        x = torch.cat((image, prev_mask, scribble), dim=1)
        l1, l2, l3, l4 = self.resnet(x)

        # Aggregation block
        agg = self.feature_aggregation(l4, prev_agg, l4.size(1))

        # Decoder
        x = self.decoder1(agg, l3)
        x = self.decoder2(x, l2)
        x = self.decoder3(x, l1)
        x = self.trans_conv(x)
        mask = F.upsample(x, scale_factor=4, mode='bilinear')
        
        return mask, agg


class PropogationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.resnet = resnet18(pretrained=True, input_layers=5)

        # Aggregation block
        self.feature_aggregation = AggregateBlock()

        # Decoder
        self.decoder1 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder3 = DecoderBlock(128, 64)
        self.trans_conv = nn.ConvTranspose2d(64, 1, kernel_size=3)

    def forward(self, image, prev_mask, prev_time_mask, interact_agg):
        # Encoder
        x = torch.cat((image, prev_mask, prev_time_mask), dim=1)
        l1, l2, l3, l4 = self.resnet(x)

        # Aggregation block
        agg = self.feature_aggregation(l4, prev_agg, l4.size(1))
        agg += interact_agg

        # Decoder
        x = self.decoder1(agg, l3)
        x = self.decoder2(x, l2)
        x = self.decoder3(x, l1)
        x = self.trans_conv(x)
        mask = F.upsample(x, scale_factor=4, mode='bilinear')
        
        return mask, agg
        

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock).__init__()
        self.residual_skip = nn.ResidualBlock(in_channels=out_channels, out_channels=in_channels)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        self.residual = nn.ResidualBlock(in_channels=in_channels, out_channels=out_channels)

    def forward(self, input, skip_connection):
        skip_input = self.residual(skip_connection)
        x = self.upsample(input)
        x = skip_input + x
        x = self.residual(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


if __name__ == '__main__':
    batch = 16
    model = ScribbleNet()
    model.cuda()
    output = model(image=torch.randn(batch, 3, 256, 256).cuda(),
                   prev_mask=torch.randn(batch, 1, 256, 256).cuda(),
                   prev_time_mask=torch.randn(batch, 1, 256, 256).cuda(),
                   scribble=torch.randn(batch, 2, 256, 256).cuda())
    print(output.shape)
