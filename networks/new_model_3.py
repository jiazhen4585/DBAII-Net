"""
(3)Ablation experiment: backbone+CMAM;
"""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class Conv_Stem(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3, 3, 3]):
        super().__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch // 4, kernel_size=kernel_size, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_ch // 4, out_ch // 2, kernel_size=kernel_size, stride=1, padding=1, bias=False)

        self.conv3 = nn.Conv3d(in_ch, out_ch // 4, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv3d(out_ch // 4, out_ch // 2, kernel_size=kernel_size, stride=2, padding=1, bias=False)

        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)

        x2 = self.conv3(x)
        x2 = self.conv4(x2)

        result = torch.cat((x1, x2), dim=1)
        result = self.bn(result)
        result = self.relu(result)

        return result


class BasicBlock(nn.Module):
    """
    Basic 3 X 3 X 3 convolution blocks.
    Extended from raghakot's 2D impl.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Basic 3 X 3 X 3 convolution blocks.
    Extended from raghakot's 2D impl.
    """

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.in_channels = in_channels

        # Define the query, key, and value projections
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x1, x2):
        # Project x1 for queries
        query = self.query_conv(x1)
        query = query.view(query.size(0), -1, query.size(2), query.size(3), query.size(4))

        # Project x2 for keys and values
        key = self.key_conv(x2)
        value = self.value_conv(x2)
        key = key.view(key.size(0), -1, key.size(2), key.size(3), key.size(4))
        value = value.view(value.size(0), -1, value.size(2), value.size(3), value.size(4))

        # Calculate attention scores
        # attn_scores = F.softmax(torch.matmul(query.permute(0, 2, 3, 4, 1), key.permute(0, 2, 3, 4, 1).transpose(-2, -1)), dim=-1)
        attn_scores = torch.matmul(query.permute(0, 2, 3, 4, 1), key.permute(0, 2, 3, 4, 1).transpose(-2, -1))

        # Apply attention to values
        weighted_values = torch.matmul(attn_scores, value.permute(0, 2, 3, 4, 1))

        # Rearrange and reshape the output
        out = weighted_values.permute(0, 4, 1, 2, 3).contiguous()
        out = out.view(out.size(0), self.in_channels, out.size(2), out.size(3), out.size(4))

        return out


class my_model(nn.Module):
    """
    ResNet3D.
    """

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv_stem = Conv_Stem(n_input_channels, self.in_planes, kernel_size=[3, 3, 3])

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)

        self.cross_attention = CrossAttention(in_channels=block_inplanes[1])

        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion // 2, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):  # x1=(2,1,64,128,128)  x2=(2,1,64,128,128)
        x1 = self.conv1(x1)    # x1=(2,64,32,64,64)
        x1 = self.bn1(x1)      #
        x1 = self.relu(x1)     #
        if not self.no_max_pool:
            x1 = self.maxpool(x1)    # x=(2,64,12,7,7)

        x2 = self.conv1(x2)  # x1=(2,64,32,64,64)
        x2 = self.bn1(x2)  #
        x2 = self.relu(x2)  #
        if not self.no_max_pool:
            x2 = self.maxpool(x2)  # x=(2,64,12,7,7)

        # x1 = self.conv_stem(x1)  # x1=(2,64,32,64,64)

        x1 = self.layer1(x1)  # x1=(2,64,32,64,64)
        x1 = self.layer2(x1)  # x1=(2,128,16,16,16)

        # x2 = self.conv_stem(x2)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)

        x = self.cross_attention(x1, x2)
        # x = torch.add(x1, x2)    # x=(2,128,16,16,16)

        x = self.layer3(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)     # x=(2,256)

        return x


def generate_model(model_depth, n_input_channels=1, n_classes=2, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = my_model(BasicBlock, [1, 1, 1, 1], get_inplanes(), n_input_channels=n_input_channels,
                         n_classes=n_classes, **kwargs)
    elif model_depth == 18:
        model = my_model(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=n_input_channels,
                         n_classes=n_classes, **kwargs)
    elif model_depth == 34:
        model = my_model(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_input_channels=n_input_channels,
                         n_classes=n_classes, **kwargs)
    elif model_depth == 50:
        model = my_model(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_input_channels=n_input_channels,
                         n_classes=n_classes, **kwargs)
    elif model_depth == 101:
        model = my_model(Bottleneck, [3, 4, 23, 3], get_inplanes(), n_input_channels=n_input_channels,
                         n_classes=n_classes, **kwargs)
    elif model_depth == 152:
        model = my_model(Bottleneck, [3, 8, 36, 3], get_inplanes(), n_input_channels=n_input_channels,
                         n_classes=n_classes, **kwargs)
    elif model_depth == 200:
        model = my_model(Bottleneck, [3, 24, 36, 3], get_inplanes(), n_input_channels=n_input_channels,
                         n_classes=n_classes, **kwargs)

    return model


if __name__ == "__main__":
    # Testing the complete model.
    model_depth = 18
    model = generate_model(model_depth, n_input_channels=1, n_classes=2)
    sample_input1 = torch.randn(2, 1, 64, 128, 128)
    sample_input2 = torch.randn(2, 1, 64, 128, 128)

    with torch.no_grad():
        model.eval()
        output = model(sample_input1, sample_input2)

    print("Output shape:", output.shape)
    print("Output:", output)

    # # Testing Conv_Stem module.
    # model = Conv_Stem(1, 64, kernel_size=[3, 3, 3])
    # sample_input = torch.randn(2, 1, 48, 28, 28)
    # with torch.no_grad():
    #     model.eval()
    #     output = model(sample_input)
    #
    # print("Output shape:", output.shape)
    # print("Output:", output)

    # # Testing the Cross Attention Module.
    # cross_attention = CrossAttention(in_channels=64)
    # x1 = torch.randn(2, 64, 64, 128, 128)
    # x2 = torch.randn(2, 64, 64, 128, 128)
    #
    # # Forward pass through the cross attention
    # output = cross_attention(x1, x2)
    #
    # # Print the output shape
    # print("Output shape:", output.shape)

