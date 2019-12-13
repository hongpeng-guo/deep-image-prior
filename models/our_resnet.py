import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm2d(planes, affine=True)
        self.relu       = nn.LeakyReLU(0.2, inplace=True)
        self.conv2      = conv3x3(planes, planes, stride)
        self.bn2        = nn.BatchNorm2d(planes, affine=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        residual = x
        out      = self.conv1(x)
        out      = self.bn1(out)
        out      = self.relu(out)
        out      = self.conv2(out)
        out      = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    # def __init__(self, basic_block, num_blocks, num_classes):
    def __init__(self, num_input_channels, num_output_channels, num_channels, pad='reflection'):
        super(ResNet, self).__init__()
        
        self.inplanes = num_channels


        self.conv1    = nn.Conv2d(num_input_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode=pad)
        #self.bn1      = nn.BatchNorm2d(num_channels, affine=True)
        self.relu     = nn.LeakyReLU(0.2, inplace=True)


        self.b1       = self._make_layer(BasicBlock, num_channels)
        self.b2       = self._make_layer(BasicBlock, num_channels)
        self.b3       = self._make_layer(BasicBlock, num_channels)
        self.b4       = self._make_layer(BasicBlock, num_channels)
        
        
        
        self.last     = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.bnl      = nn.BatchNorm2d(num_channels, affine=True)
        self.out      = nn.Conv2d(num_channels, num_output_channels, 3, 1, 1, bias=True, padding_mode=pad)
        self.sig      = nn.Sigmoid()







    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)

        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        
        x = self.last(x)
        x = self.bnl(x)
        x = self.out(x)
        x = self.sig(x)


        return x
    
    
    def _make_layer(self, block, planes, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        #for _ in range(1, blocks):
        layers.append(block(self.inplanes, planes, stride))

        return nn.Sequential(*layers)
