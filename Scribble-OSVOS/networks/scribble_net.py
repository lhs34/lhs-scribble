import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from networks.resnet import *


class ScribbleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.interaction = InteractionNetwork()
        self.propogation = PropogationNetwork()

    def forward(self, inputs, scribbles, scribble_idx, prev_masks, prev_agg):
        mask_scribble_idx, agg = self.interaction(inputs[scribble_idx], prev_masks[scribble_idx], scribbles[-1], prev_agg)
        video_len = inputs.shape[0]
        masks = []

        # Propogate forwards in video sequence
        prev_time_mask = mask_scribble_idx
        masks.append(prev_time_mask)
        for i in range(scribble_idx, video_len):
            prev_time_mask = self.propogation(inputs[scribble_idx], prev_masks[scribble_idx], prev_time_mask, prev_agg)
            masks.append(prev_time_mask)
        
        # Go backwards in video sequence
        prev_time_mask = mask_scribble_idx
        for i in range(scribble_idx-1, -1, -1):
            prev_time_mask = self.propogation(inputs[scribble_idx], prev_masks[scribble_idx], prev_time_mask, prev_agg)
            masks = [prev_time_mask] + masks
        
        masks = torch.stack(masks)
        return masks, agg


class InteractionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.resnet = resnet18(pretrained=True, input_layers=6)

        # Aggregation block
        self.feature_aggregation = AggregateBlock(512, (15,26))

        # Decoder
        self.decoder1 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder3 = DecoderBlock(128, 64)
        self.trans_conv = nn.ConvTranspose2d(64, 1, kernel_size=1)

    def forward(self, image, prev_mask, scribble, prev_agg):
        # Encoder
        x = torch.cat((image, prev_mask, scribble), dim=1)
        l1, l2, l3, l4 = self.resnet(x)

        # Aggregation block
        agg = self.feature_aggregation(prev_agg, l4)

        # Decoder
        x = self.decoder1(l4, l3)
        x = self.decoder2(x, l2)
        x = self.decoder3(x, l1)
        x = self.trans_conv(x)
        mask = F.interpolate(x, scale_factor=4, mode='bilinear')
        return mask, agg


class PropogationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.resnet = resnet18(pretrained=True, input_layers=5)

        # Decoder
        self.decoder1 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder3 = DecoderBlock(128, 64)
        self.trans_conv = nn.ConvTranspose2d(64, 1, kernel_size=1)

    def forward(self, image, prev_mask, prev_time_mask, interact_agg):
        # Encoder
        x = torch.cat((image, prev_mask, prev_time_mask), dim=1)
        l1, l2, l3, l4 = self.resnet(x)

        # Feature fusion
        agg = torch.cat((l4, interact_agg), dim=1)

        # Decoder
        x = self.decoder1(agg, l3)
        x = self.decoder2(x, l2)
        x = self.decoder3(x, l1)
        x = self.trans_conv(x)
        mask = F.interpolate(x, scale_factor=4, mode='bilinear')
        return mask


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual_skip = ResidualBlock(out_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels,1)
        self.residual = ResidualBlock(out_channels, out_channels)

    def forward(self, input, skip_connection):
        skip_input = self.residual_skip(skip_connection)
        x = F.interpolate(input, scale_factor=2,mode='bilinear')
        x = self.conv(x)
        x = skip_input + x
        x = self.residual(x)
        return x


class AggregateBlock(nn.Module):
    def __init__(self, out_channels, size):
        super().__init__()
        self.out_channels = out_channels
        self.size = size
        self.gap_prev = nn.AvgPool2d(size)
        self.gap_curr =  nn.AvgPool2d(size)
        self.downsample = nn.Linear( 2*out_channels, out_channels  )
        self.upsample = nn.Linear( out_channels, 2*out_channels  )

    def forward(self, prev_map, curr_map):
        prev_map_ga_pool = self.gap_prev(prev_map).view(-1, self.out_channels )
        curr_map_ga_pool = self.gap_curr(curr_map).view(-1, self.out_channels )
        x = torch.cat( (prev_map_ga_pool, curr_map_ga_pool), dim=1)
        x = self.downsample(x)
        x = self.upsample(x)
        x =  x.view(x.shape[0], -1, 2 )
        x = torch.softmax(x, dim = -1)
        weighted_prev_map_ga_pool =  torch.mul( prev_map.view(x.shape[0], self.out_channels, -1),   x[:, :, 0].unsqueeze(-1) )
        weighted_curr_map_ga_pool =  torch.mul( curr_map.view(x.shape[0], self.out_channels, -1),   x[:, :, 1].unsqueeze(-1) )
        out = weighted_prev_map_ga_pool.view(x.shape[0], self.out_channels, self.size[0], self.size[1]) + weighted_curr_map_ga_pool.view(x.shape[0], self.out_channels, self.size[0], self.size[1])
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
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


# if __name__ == '__main__':
#     batch = 16
#     model = ScribbleNet()
#     model.cuda()
#     output = model(image=torch.randn(batch, 3, 256, 256).cuda(),
#                    prev_mask=torch.randn(batch, 1, 256, 256).cuda(),
#                    prev_time_mask=torch.randn(batch, 1, 256, 256).cuda(),
#                    scribble=torch.randn(batch, 2, 256, 256).cuda(),
#                    prev_agg = torch.randn(batch, 512, 8, 8).cuda())
#     print(output[0].shape)
#     print(output[1].shape)
