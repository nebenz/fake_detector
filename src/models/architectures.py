import sys
import torch.nn as nn
import torch
import numpy as np
# from parts_model import *
# import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.utils import weight_norm
import math
import torchvision.models as models



epsilon = 1e-6

class Simple_Classification(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.conv1 = nn.Conv2d(1, 128, kernel_size=(10,1))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        #TODO dimensions are hard coded!
        self.fc1 = nn.Linear(26912, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,1)

        self.sigmoid = nn.Sigmoid()
      
    
    def forward(self, x):

        x = torch.log10(x+epsilon) 
        x =  self.pool1(torch.relu(self.conv1(x)))
        x =  self.pool2(torch.relu(self.conv2(x)))
        x =  self.pool3(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)

    
        return x

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, args):
        super(ResNetBinaryClassifier, self).__init__()
        self.args= args
        # Load pre-trained ResNet model
        import ssl
        import urllib.request
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        self.resnet = models.resnet18(pretrained=True)
        
        # first layer to accept single channel 
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace last layer with binary classification layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x = torch.log10(x+epsilon) 
        x = self.resnet(x)
        x = self.sigmoid(x)

        return x





class Encoder_Classification(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.inc = DoubleConv2d(1,64)
        self.down1 = Down(64, 128,reduce_temporal=1)
        self.down2 = Down(128, 256,reduce_temporal=1)
        self.down3 = Down(256, 512,reduce_temporal=1)
        self.down4 = Down(512, 512,reduce_temporal=1)

        self.down5 = nn.Linear(8192, 512)
        
    

        self.encod1 = nn.TransformerEncoderLayer(d_model=512, nhead=16,  batch_first=True, dropout=0.1, dim_feedforward=2048)
        self.encod2 = nn.TransformerEncoderLayer(d_model=512, nhead=16,  batch_first=True, dropout=0.1, dim_feedforward=2048)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)


        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        x = torch.log10(x+epsilon) 
       
        x = self.inc(x)
        x  = self.down1(x)
        x  = self.down2(x)
        x  = self.down3(x)
        x  = self.down4(x) #
        x  = x.flatten(start_dim=1, end_dim=2).transpose(1,2)

        x = self.down5(x)

        x= self.encod1(x)
        x= self.encod2(x)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.sigmoid(x)
        x = torch.mean(x,axis=1)

        return x
       


class UNet_R_with_transformer_mask_log(nn.Module):
    def __init__(self,bilinear=True):
        super(UNet_R_with_transformer_mask_log,self).__init__()
        self.bilinear = bilinear
        self.inc = DoubleConv2d(1,64)
        self.down1 = Down(64, 128,reduce_temporal=1)
        self.down2 = Down(128, 256,reduce_temporal=1)
        self.down3 = Down(256, 512,reduce_temporal=1)
        self.down4 = Down(512, 512,reduce_temporal=1)

        self.down5 = nn.Linear(8192, 512)
        self.up0 = nn.Linear(512, 4096)


        self.up1 = Up(1024, 256, bilinear,reduce_temporal=1)
        self.up2 = Up(512, 128, bilinear,reduce_temporal=1)
        self.up3 = Up(256, 64, bilinear,reduce_temporal=1)
        self.up4 = Up(128, 32, bilinear,reduce_temporal=1)

        #LOG - Tan Hyp! 
        self.outc = OutConv2d_Tanh(32, 1)

        self.encod1 = nn.TransformerEncoderLayer(d_model=512, nhead=16,  batch_first=True, dropout=0.1, dim_feedforward=2048)
        self.encod2 = nn.TransformerEncoderLayer(d_model=512, nhead=16,  batch_first=True, dropout=0.1, dim_feedforward=2048)
        
        # self.mu = nn.Linear(512, 1)
        self.max = nn.Linear(512, 1)
        
    def forward(self, batch):
        input_logspec = torch.log10(batch[0]+epsilon) #min max -4,  2
        x1 = self.inc(input_logspec)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # -2 , 13
        x6 = x5.flatten(start_dim=1, end_dim=2).transpose(1,2)

        x7 = self.down5(x6)

        x8= self.encod1(x7)
        x9= self.encod2(x8)

        x10 = self.up0(x9) # -4, 4
        x11 = x10.transpose(1,2).view(-1,512,8,x10.shape[-2])

        x = self.up1(x11, x4) # x.shape 256,32,256   # 0, 8
        x = self.up2(x, x3) # x.shape 128,64,256
        x = self.up3(x, x2) # x.shape 64,128,256
        x = self.up4(x, x1) # x.shape 32,256,256    # -0.4 ,  5
        logits = self.outc(x) # 1,256,256    #   -0.0001, 0.9
        #for masking:
        pred = logits*batch[0]   # -0.0007,   2.8
        #pred = logits*input_logspec

         
        x8_average = torch.mean(x8,axis=1)
        
        # x8_view = x8_average.view(x8_average.size(0),-1)
        # self.learned_mean = self.mu(x8_average).view(-1,1,1,1)
        self.learned_max = self.max(x8_average).view(-1,1,1,1)   # 20
        # unet_output =  (logits*self.learned_max)   -0.013,   56
        unet_output = (pred*self.learned_max)
        return unet_output ,logits



class UNet_R_with_transformer_map_log(nn.Module):
    def __init__(self,bilinear=True):
        super(UNet_R_with_transformer_map_log,self).__init__()
        self.bilinear = bilinear
        self.inc = DoubleConv2d(1,64)
        self.down1 = Down(64, 128,reduce_temporal=3)
        self.down2 = Down(128, 256,reduce_temporal=1)
        self.down3 = Down(256, 512,reduce_temporal=1)
        self.down4 = Down(512, 512,reduce_temporal=1)

        self.down5 = nn.Linear(8192, 512)
        self.up0 = nn.Linear(512, 4096)


        self.up1 = Up(1024, 256, bilinear,reduce_temporal=1)
        self.up2 = Up(512, 128, bilinear,reduce_temporal=1)
        self.up3 = Up(256, 64, bilinear,reduce_temporal=1)
        self.up4 = Up(128, 32, bilinear,reduce_temporal=1)

        
        #self.outc = OutConv2d(32, 1)
        self.outc = OutConv2d_Tanh(32, 1)

        self.encod1 = nn.TransformerEncoderLayer(d_model=512, nhead=16,  batch_first=True, dropout=0.1, dim_feedforward=2048)
        self.encod2 = nn.TransformerEncoderLayer(d_model=512, nhead=16,  batch_first=True, dropout=0.1, dim_feedforward=2048)
        
        # self.mu = nn.Linear(512, 1)
        self.max = nn.Linear(512, 1)
        
    def forward(self, batch):
        input_logspec = torch.log10(batch[0]+epsilon)
        x1 = self.inc(input_logspec)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = x5.flatten(start_dim=1, end_dim=2).transpose(1,2)

        x7 = self.down5(x6)

        x8= self.encod1(x7)
        x9= self.encod2(x8)

        x10 = self.up0(x9)
        x11 = x10.transpose(1,2).view(-1,512,8,x10.shape[-2])

        x = self.up1(x11, x4) # x.shape 256,32,256
        x = self.up2(x, x3) # x.shape 128,64,256
        x = self.up3(x, x2) # x.shape 64,128,256
        x = self.up4(x, x1) # x.shape 32,256,256
        logits = self.outc(x) # 1,256,256
        
        #for mapping:
        pred = logits  # shouldn't need to be log10(batch[0])?
        
        
        x8_average = torch.mean(x8,axis=1)
        
        # x8_view = x8_average.view(x8_average.size(0),-1)
        # self.learned_mean = self.mu(x8_average).view(-1,1,1,1)
        self.learned_max = self.max(x8_average).view(-1,1,1,1)
        # unet_output =  (logits*self.learned_max)
        unet_output = (pred*self.learned_max)
        #conver to abs

        return unet_output ,logits





# KERNEL_SIZE = 3


class DoubleConv2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
# TODO : change to second kernel size - 3
    def __init__(self,in_channels, out_channels,mid_channels=None,dilation=1,first_kernal_size=3,second_kernal_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if dilation>1:
            num_ReflectionPad1d = dilation
        else:
            num_ReflectionPad1d = int(first_kernal_size/2)
            
        self.double_conv = nn.Sequential(
            nn.ReflectionPad2d(num_ReflectionPad1d),
            weight_norm(nn.Conv2d(in_channels,mid_channels,kernel_size=first_kernal_size,dilation=dilation)),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
            nn.ReflectionPad2d(int(second_kernal_size/2)),
            weight_norm(nn.Conv2d(mid_channels,out_channels,kernel_size=second_kernal_size)),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self,in_channels, out_channels,reduce_temporal = 2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((2,reduce_temporal)),
            DoubleConv2d(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
#!#
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self,in_channels, out_channels, bilinear=True,reduce_temporal = 2):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=(2,reduce_temporal), mode='bilinear', align_corners=True)
            self.conv = DoubleConv2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2d(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv2d(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(OutConv2d, self).__init__()
        self.conv =nn.Sequential(
        weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)),
        nn.Sigmoid()
        )
    def forward(self, x):
        return self.conv(x)

class OutConv2d_Tanh(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(OutConv2d_Tanh, self).__init__()
        self.conv =nn.Sequential(
        weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)),
        nn.Tanh()
        )
    def forward(self, x):
        return self.conv(x)



