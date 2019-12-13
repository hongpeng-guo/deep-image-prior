import torch
import torch.nn as nn
from .common import *
from .our_endec import Downsample, Upsample, UpsampleNoSkip

def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], 
        filter_size_down=3, upsample_mode='bilinear'):
    
    if len(num_channels_up) < 2:
        raise ValueError("Depth should be more than 2!")
    
    
    if len(num_channels_down) != len(num_channels_up):
        raise ValueError("Upsampling and Downsampling channels should be of the same size!")
    
    
    n_scales = len(num_channels_down)
    
    model = nn.Sequential()
    cur_model = model
    
    input_size = num_input_channels
    
    for i in range(n_scales):
        
        new_layer = nn.Sequential()
        cur_model.add(new_layer)
        
        new_layer.add(    Downsample(input_size, num_channels_down[i], filter_size_down)   )
        
        next_layers = nn.Sequential()
        
        if i == n_scales - 1:
            up_input_size = num_channels_down[i]
        else:
            new_layer.add(next_layers)
            up_input_size = num_channels_up[i+1]
        
        new_layer.add(   UpsampleNoSkip(up_input_size, num_channels_up[i], upsample_mode=upsample_mode)  )
        
        input_size = num_channels_down[i]
        
        cur_model = next_layers
    model.add(nn.Conv2d(num_channels_up[0], num_output_channels, 1, bias=True))
    model.add(nn.Sigmoid())
    return model
        
    
    
    