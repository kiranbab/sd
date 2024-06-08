import torch 
from torch import nn
from torch.nn import Sequential as F 
from attention import SelfAttention,CrossAttention 

class TimeEmbedding(nn.Module):
    
    def __init__(self,n_embd:int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd,4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd,4*n_embd)
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # x :(1,320)
        
        x=self.linear(x)
        
        x= F.silu(x)
        
        x=self.linear_2(x)
        
        # (1,1280)
        return x 


class Diffusion(nn.Module):
    
    def __init__(self):
        self.time_embedding= TimeEmbedding(320)
        self.unet = UNET() 
        self.final = UNET_OutputLayer(320,4)
        
    def forward(self,latent:torch.Tensor,context: torch.Tensor,time: torch.Tensor):
        # latent :(batch_size,4,height/8,width/8) -> output of encoder
        # context:(batch_zie,seq_len,dim) -> clip output
        # time : (1,320)
        
        # (1,320) -> (1,1280)
        time = self.time_embedding(time)
        
        # (batch_size,4,height/8,width/8) -> (batch_size,320,height/8,width/8)
        output = self.unet(latent,context,time)
        
        # (batch_size,320,height/8,width/8) -> (batch_size,4,height/8,width/8)
        output = self.final(output)
        
        # (batch_size,4,height/8,width/8)
        return output