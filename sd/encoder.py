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
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0)
            
            # we just increase the number of features here 
            VAE_ResidualBlock(128,256),
            
            VAE_ResidualBlock(256,256),
        )