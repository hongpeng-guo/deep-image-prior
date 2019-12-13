import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *
from torch.nn import init

    
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_kernel_size = 3):
        super().__init__()
        #self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 5, stride = 2, padding = 2, bias = True, padding_mode='reflection')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = downsample_kernel_size, stride = 2, padding = int((downsample_kernel_size-1)/2), bias = True, padding_mode='reflection')
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        #self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 5, stride = 1, padding = 2, bias = True, padding_mode='reflection')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = downsample_kernel_size, stride = 1, padding = int((downsample_kernel_size-1)/2), bias = True, padding_mode='reflection')
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, upsample_mode='bilinear', need1x1_up=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode=upsample_mode)#, align_corners=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True, padding_mode='reflection')
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True, padding_mode='reflection')
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.need1x1 = need1x1_up



    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        
        #x = self.up(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.need1x1:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
        
        
        return x

    
class UpsampleNoSkip(nn.Module):

    def __init__(self, in_channels, out_channels, upsample_mode='bilinear'):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode=upsample_mode)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True, padding_mode='reflection')
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True, padding_mode='reflection')
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)



    def forward(self, x):
        
        x = self.up(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        '''
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        '''
        
        
        
        return x

class Conn(nn.Module):
    def __init__(self, in_channels, out_channels, filter_skip_size):
        super(Conn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=filter_skip_size, bias=True)
        self.bn    = nn.BatchNorm2d(out_channels)
        self.relu  = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    


class EnDec(nn.Module):
    
    def __init__(
        self, num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128, 128], num_channels_up=[16, 32, 64, 128, 128, 128], num_channels_skip=[4, 4, 4, 4, 4, 4], 
        filter_size_down=3, filter_skip_size=1, upsample_mode='nearest', need1x1_up=False):
        
        super(EnDec, self).__init__()
        
        if len(num_channels_up) < 2:
            raise ValueError("Depth should be more than 2!")
        
        if len(num_channels_down) != len(num_channels_up) or len(num_channels_up) != len(num_channels_skip):
            raise ValueError("Upsampling, Downsampling, and Skip channels should be of the same size!")
        
        n_scales = len(num_channels_down)
        

        self.need_skip = num_channels_skip[0] != 0
        self.need1x1_up = need1x1_up

        
        # depth 5
        '''
        self.down1 = Downsample(num_input_channels, num_channels_down[0], downsample_kernel_size=filter_size_down)
        self.down2 = Downsample(num_channels_down[0], num_channels_down[1], downsample_kernel_size=filter_size_down)
        self.down3 = Downsample(num_channels_down[1], num_channels_down[2], downsample_kernel_size=filter_size_down)
        self.down4 = Downsample(num_channels_down[2], num_channels_down[3], downsample_kernel_size=filter_size_down)
        self.down5 = Downsample(num_channels_down[3], num_channels_down[4], downsample_kernel_size=filter_size_down)
        
        if self.need_skip:
            self.skip1 = Conn(num_input_channels, num_channels_skip[0], filter_skip_size)
            self.skip2 = Conn(num_channels_down[0], num_channels_skip[1], filter_skip_size)
            self.skip3 = Conn(num_channels_down[1], num_channels_skip[2], filter_skip_size)
            self.skip4 = Conn(num_channels_down[2], num_channels_skip[3], filter_skip_size)
            self.skip5 = Conn(num_channels_down[3], num_channels_skip[4], filter_skip_size)        

            self.up1   = Upsample(num_channels_skip[4] + num_channels_down[4], num_channels_up[4], upsample_mode, self.need1x1_up)
            self.up2   = Upsample(num_channels_skip[3] + num_channels_up[4], num_channels_up[3], upsample_mode, self.need1x1_up)
            self.up3   = Upsample(num_channels_skip[2] + num_channels_up[3], num_channels_up[2], upsample_mode, self.need1x1_up)
            self.up4   = Upsample(num_channels_skip[1] + num_channels_up[2], num_channels_up[1], upsample_mode, self.need1x1_up)
            self.up5   = Upsample(num_channels_skip[0] + num_channels_up[1], num_channels_up[0], upsample_mode, self.need1x1_up)               
            
        else:

            self.up1   = UpsampleNoSkip(num_channels_down[4], num_channels_up[4], upsample_mode)
            self.up2   = Upsample(num_channels_up[4] + num_channels_down[3], num_channels_up[3], upsample_mode)
            self.up3   = Upsample(num_channels_up[3] + num_channels_down[2], num_channels_up[2], upsample_mode)
            self.up4   = Upsample(num_channels_up[2] + num_channels_down[1], num_channels_up[1], upsample_mode)
            self.up5   = Upsample(num_channels_up[1] + num_channels_down[0], num_channels_up[0], upsample_mode)
        # depth 5 end
                        
            
        '''    
        # depth 6  
        self.down1 = Downsample(num_input_channels, num_channels_down[0], downsample_kernel_size=filter_size_down)
        self.down2 = Downsample(num_channels_down[0], num_channels_down[1], downsample_kernel_size=filter_size_down)
        self.down3 = Downsample(num_channels_down[1], num_channels_down[2], downsample_kernel_size=filter_size_down)
        self.down4 = Downsample(num_channels_down[2], num_channels_down[3], downsample_kernel_size=filter_size_down)
        self.down5 = Downsample(num_channels_down[3], num_channels_down[4], downsample_kernel_size=filter_size_down)
        self.down6 = Downsample(num_channels_down[4], num_channels_down[5], downsample_kernel_size=filter_size_down)
        
        if self.need_skip:
            self.skip1 = Conn(num_input_channels, num_channels_skip[0], filter_skip_size)
            self.skip2 = Conn(num_channels_down[0], num_channels_skip[1], filter_skip_size)
            self.skip3 = Conn(num_channels_down[1], num_channels_skip[2], filter_skip_size)
            self.skip4 = Conn(num_channels_down[2], num_channels_skip[3], filter_skip_size)
            self.skip5 = Conn(num_channels_down[3], num_channels_skip[4], filter_skip_size)        
            self.skip6 = Conn(num_channels_down[4], num_channels_skip[5], filter_skip_size)        

            self.up1   = Upsample(num_channels_skip[5] + num_channels_down[5], num_channels_up[5], upsample_mode, self.need1x1_up)
            self.up2   = Upsample(num_channels_skip[4] + num_channels_up[5], num_channels_up[4], upsample_mode, self.need1x1_up)
            self.up3   = Upsample(num_channels_skip[3] + num_channels_up[4], num_channels_up[3], upsample_mode, self.need1x1_up)
            self.up4   = Upsample(num_channels_skip[2] + num_channels_up[3], num_channels_up[2], upsample_mode, self.need1x1_up)
            self.up5   = Upsample(num_channels_skip[1] + num_channels_up[2], num_channels_up[1], upsample_mode, self.need1x1_up)
            self.up6   = Upsample(num_channels_skip[0] + num_channels_up[1], num_channels_up[0], upsample_mode, self.need1x1_up)   
            
        else:
            self.up1   = UpsampleNoSkip(num_channels_down[5], num_channels_up[5], upsample_mode)
            self.up2   = Upsample(num_channels_up[5] + num_channels_down[4], num_channels_up[4], upsample_mode)
            self.up3   = Upsample(num_channels_up[4] + num_channels_down[3], num_channels_up[3], upsample_mode)
            self.up4   = Upsample(num_channels_up[3] + num_channels_down[2], num_channels_up[2], upsample_mode)
            self.up5   = Upsample(num_channels_up[2] + num_channels_down[1], num_channels_up[1], upsample_mode)
            self.up6   = Upsample(num_channels_up[1] + num_channels_down[0], num_channels_up[0], upsample_mode)
            self.up6   = UpsampleNoSkip(num_channels_up[0], num_channels_up[0], upsample_mode)
        # depth 6 end
            
            
            
            
        self.outconv = nn.Conv2d(num_channels_up[0], num_output_channels, 1, bias=True)
        self.sig     = nn.Sigmoid()
        

    
    
    def forward(self, x):

        
        # depth 5
        '''
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        if self.need_skip:
            s1 = self.skip1(x)
            s2 = self.skip2(x1)
            s3 = self.skip3(x2)
            s4 = self.skip4(x3)
            s5 = self.skip5(x4)

            u = self.up1( x5, s5 )
            u = self.up2( u,  s4 )        
            u = self.up3( u,  s3 )
            u = self.up4( u,  s2 )
            u = self.up5( u,  s1 )
        else:  
            u = self.up1(x6)
            u = self.up2(u)
            u = self.up3(u)
            u = self.up4(u)
            u = self.up5(u)
            u = self.up6(u)
        '''    
            
        # depth 6    
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        if self.need_skip:
            s1 = self.skip1(x)
            s2 = self.skip2(x1)
            s3 = self.skip3(x2)
            s4 = self.skip4(x3)
            s5 = self.skip5(x4)
            s6 = self.skip6(x5)
            
            u = self.up1( x6, s6 )
            u = self.up2( u,  s5 )
            u = self.up3( u,  s4 )        
            u = self.up4( u,  s3 )
            u = self.up5( u,  s2 )
            u = self.up6( u,  s1 )

        else:
            u = self.up1(x6)
            u = self.up2(u,  x5)
            u = self.up3(u,  x4)
            u = self.up4(u,  x3)
            u = self.up5(u,  x2)
            u = self.up6(u,  x1)
        
        return self.sig(self.outconv(u))
                
        
        
        
        
        
        
        