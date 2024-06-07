import torch 
from torch import nn 
from torch.nn import functional as F 
from decoder import VAE_AttentionBlock,VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init(
            # intially channel size is 3 denoting rgb
            # (batch size, channel, height,width) -> (batch_size,128,height,width)
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            VAE_ResidualBlock(128,128),
            # residual block is bunch of convolutions and normalizations and doesn't change the size 
            # (batchsize,128,height,width) ->(batchsize,128,height,width)
            VAE_ResidualBlock(128,128),
            # (batchsize,128,height,width) ->(batchsize,128,height,width)
            
            # (batchsize,128,height,width) ->(batchsize,128,height/2,width/2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            # (batchsize,128,height/2,width/2) ->(batchsize,256,height/2,width/2)
            VAE_ResidualBlock(128,256),
            # (batchsize,256,height/2,width/2)->(batchsize,256,height/2,width/2)
            VAE_ResidualBlock(256,256),
            
            # (batchsize,256,height/2,width/2) ->(batchsize,256,height/4,width/4)
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            # (batchsize,256,height/4,width/4) ->(batchsize,512,height/4,width/4)
            VAE_ResidualBlock(256,512),
            # (batchsize,512,height/4,width/4)->(batchsize,512,height/4,width/4)
            VAE_ResidualBlock(512,512),

            # (batchsize,512,height/2,width/2) ->(batchsize,512,height/8,width/8)
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
            # (batchsize,512,height/8,width/8) ->(batchsize,512,height/8,width/8)
            VAE_ResidualBlock(512,512),
            # (batchsize,512,height/8,width/8) ->(batchsize,512,height/8,width/8)
            VAE_ResidualBlock(512,512),
            # (batchsize,512,height/8,width/8) ->(batchsize,512,height/8,width/8)
            VAE_ResidualBlock(512,512),
            
            # running a attention mechanism over each layer of image and there is no change in shape
            # (batchsize,512,height/8,width/8) ->(batchsize,512,height/8,width/8)
            VAE_AttentionBlock(512), 
            
            # (batchsize,512,height/8,width/8) ->(batchsize,512,height/8,width/8)
            VAE_ResidualBlock(512,512),
            # apply group normaliations
            # (batchsize,512,height/8,width/8) ->(batchsize,512,height/8,width/8)
            nn.GroupNorm(32,412),
            # (batchsize,512,height/8,width/8) ->(batchsize,512,height/8,width/8)
            nn.SiLU(),
            
            # (batchsize,512,height/8,width/8) ->(batchsize,8,height/8,width/8)
            nn.Conv2d(512,8),
            # (batchsize,8,height/8,width/8) ->(batchsize,8,height/8,width/8)
            nn.Conv2d(8,8,kernel_size=1,padding=0)
        )
        
    def forward(self,x:torch.Tensor,noise:torch.Tensor) -> torch.Tensor:
        # x: (batch_size,in_channel,height,width) 
        # noise:(batch_size,out_channels,height/8,width/8)
        for module in self:
            if getattr(module,'stride',None)==(2,2):
                # deafult padding is symmetrically applied on left,right,top,bottom but we want asymetric padding on right and bottom alone 
                # (padding_left,padding_right,padding_top,padding_bottom)
                x=F.pad(x,(0,1,0,1))
            x=module(x)
            