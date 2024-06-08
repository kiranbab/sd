import torch 
from torch import nn 
from torch.nn import Sequential as F 
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    
    
    def __init__(self,channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(1,channels)
        
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # x:(Batchsize,features,height,width)
        
        residue=x 
        
        n,c,h,w=x.shape
        
        # (batchsize,features,height,width) -> (batchsize,features,height*width) 
        x=x.view(n,c,h*w)
        
        # (batchsize,features,height*width) -> (batchsize,height*width,features)
        x=x.transpose(-1,-2)
        
        # now we try to perform self attention with each pixel(height*width) with the corresponding features
        # (batchsize,height*width,features)-> (batchsize,height*width,features)
        x=self.attention(x)
        
        # (batchsize,height*width,features) -> (batchsize,features,height*width)
        x= x.transpose(-1,-2)
        
        # (batchsize,features,height*width) -> (batchsize,features,height,width) 
        x=x.view(n,c,h, w)
        
        x+=residue
        
        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.groupnorm_1=nn.GroupNorm(32,in_channels)
        self.conv_1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        
        self.groupnorm_2=nn.GroupNorm(32,out_channels)
        self.conv_2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        
        if in_channels == out_channels:
            self.residual_layer=nn.Identity() 
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
            
            
    def forward(self,x:torch.Tensor) -> torch.Tensor: 
        # x (batch_size,in_channels,height,width)
        
        residue=x 
        
        x= self.groupnorm_1(x)
        x=F.silu(x)
        
        x=self.conv_1(x)
        x=self.groupnorm_2(x)
        
        x=F.silu(x)
        x=self.conv_2(x)
        
        return x + self.residual_layer(residue)
    
class VAE_Decoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1,padding=0),
            nn.Conv2d(4,512,kernel_size=3,padding=1),
            
            VAE_ResidualBlock(512,512),
            
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            # (batchsize,512,height/8,width/8) -> (batchsize,512,height/8,width/8)
            VAE_ResidualBlock(512,512),
            
            nn.Upsample(scale_factor=2),
            
        )
         