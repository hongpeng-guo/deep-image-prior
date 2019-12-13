from .our_endec import EnDec
from .our_skip import skip
from .our_resnet import ResNet
import torch.nn as nn

def get_net(NET_TYPE, input_depth, n_channels=3):
    if NET_TYPE == 'ResNet':
        # TODO
        net = ResNet(input_depth, n_channels, 32)
    elif NET_TYPE == 'skip':
        net = skip(input_depth, n_channels)

    elif NET_TYPE == 'UNet':
        net = EnDec(input_depth, n_channels)

    else:
        assert False

    return net
