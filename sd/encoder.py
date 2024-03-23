import torch 
from torch import nn 
from torch.nn import functional as F 

from decoder import VAE_AttentionBlock, VAE_ResidualBlock 

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            
            # SInce residual block doesn't change the output size, both input and output are 128 channels depth 
            VAE_ResidualBlock(128,128),
            
            VAE_ResidualBlock(128,128),
            
            # (batchsize,128,height,width) -> (batchsize,128,height/2,width/2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            
            # we just increase the number of features here 
            VAE_ResidualBlock(128,256),
            
            VAE_ResidualBlock(256,256),
            
            # (batch_size,256,height/2,width/2) --> (batch_size,256,height/4,width/4)
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            
            # we just increase the number of features here 
            VAE_ResidualBlock(256,512),
            
            VAE_ResidualBlock(512,512),
            
            # (batch_size,512,height/4,width/4) --> (batch_size,512,height/8,width/8)
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
            
            # we just increase the number of features here 
            VAE_ResidualBlock(512,512),
            
            VAE_ResidualBlock(512,512),
            
            VAE_ResidualBlock(512,512),
            
            #size doesn't change in attention   
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512,512),
            
            # group norm doesn't change size 
            
            # (batch_size,512,height/8,width/8) --> (batch_size,512,height/8,width/8)
            nn.GroupNorm(32,512),
            
            nn.SiLU(),
            
            # bottleneck fo encoder
            
            # (batch_size,256,height/8,width/8) --> (batch_size,8,height/8,width/8)
            nn.Conv2D(512,8,kernel_size=3,padding=1),
            
            
            # (batch_size,8,height/8,width/8) --> (batch_size,8,height/8,width/8)
            nn.Conv2D(8,8,kernel_size=1,padding=0)
        )
        
    def forward(self,x:torch.Tensor,noise:torch.Tensor) -> torch.Tensor:
        # x:(batch_size,channel,height,width)
        # noise : (batch_size,output_channels,height/8,width/8)
        
        for module in self:
            if getattr(module,'stride',None)==(2,2):
                # where the stride = 2 we have padding =0 but we want assymetric padding, where we pad only in right and bottom side 
                # (padding_left,padding_right,padding_up,padding_down)
                x = F.pad(x,(0,1,0,1))
            x=module(x)
        
        #(batchsize,8,height/8,width/8) -> (batchsize,4,height/8,width/8) 
        mean,log_variance=torch.clamp(x,2,dim=1)
        
        # this doesn't change the shape of the array but only controls the variance inside the threshold values
        log_variance=torch.clamp(log_variance,-30,20)
        
        # we are removing the log by rasiing it to exponential, this doesn;t alter the shape
        variance = log_variance.exp()