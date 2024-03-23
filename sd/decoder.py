import torch 
from torch import nn 
from torch.nn import functional as F 
from attention import SelfAttention 



class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.groupnorm=nn.GroupNorm(32,channels)
        self.attention =SelfAttention(1,channels)
        
    def forward(self,x:torch.Tensor) -> torch.Tensor: 
        # x :(batch_size,channels,height,width)
        residue =x 
        n,c,h,w = x.shape
        
        # we are going to apply attention between all the pixels in this image 
        # (batch_size,channels,height,width) -> (batch_size,channels,height*width)
        # because we are mutliplying the height and width we are getting all the pixel values 
        x=x.view(n,c,h* w)
        
        # (batch_size,channels,height*width) -> (batch_size,height*width,channels)
        x=x.transpose(-1,-2)
        
        
        # we are doing since we want to do attention between each pixel and the channels(also called features)
        x=self.attention(x)
        
        # (batch_size,height*width,channels) -> (batch_size,channels,height*width)
        x=x.transpose(-1,-2)
        
         # (batch_size,channels,height*width) -> (batch_size,channels,height,width)
        x=x.view((n,c,h,w))
        
        x+=residue
        
        return x 
    
    
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1=nn.GroupNorm(32,in_channels)
        self.conv_1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        
        self.groupnorm_2=nn.GroupNorm(32,out_channels)
        self.conv_2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        
        
        if in_channels==out_channels:
            self.residual_layer=nn.Identity()
            
        else:
            # when line 41 comes when the input doesn't match the output dimension we perform this convolution to convert the input and output to same dimension 
            self.residual_layer=nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
            
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # x:(Batch_size,In_channels,height,width)
        residue=x 
        
        # this doesn't change the dimension of x 
        x=self.groupnorm_1(x)
        
        x =F.silu(x)
        
        # this doesn't change the dimensin of x even if the kernel_size is 3 since padding is 1, it return back to original shape
        x=self.conv_1(x)
        
        x=self.groupnorm_2(x)
        
        x= F.silu(x)
        
        x=self.conv_2(x)
        
        return x +self.residual_layer(residue)