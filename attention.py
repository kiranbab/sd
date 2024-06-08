import torch 
from torch import nn 
from torch.nn import functional as F 
import math 

class SelfAttention(nn.Module):
    
    def __init__(self,n_heads:int,d_embed:int,in_proj_bias=True,out_proj_bias=True):
        
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed,3*d_embed,bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_proj_bias)
        self.n_heads= n_heads
        self.d_head =d_embed// n_heads
        
    def forward(self,x:torch.Tensor,causal_mask=False):
        # x:(batch_size,seq_len,dim)
        
        input_shape=x.shape
        batch_size,sequence_length,d_embed= input_shape
        
        intermim_shape = (batch_size,sequence_length,self.n_heads,self.d_head)
        
        # (Batch_size,Seq_len,Dim) -> (Batch_size,Seq_len,Dim*3) -> 3 tensor of dime (batch_szie,seq_len,dim)
        q,k,v = self.in_proj(x).chunks(3,dim=-1)
        
        # (batch_size,seq_len,dim) -> (batch_size,seq_len,H,Dim/H) -> (batch_size,H, Seq_len,Dim/H)
        q=q.view(intermim_shape).transpose(1,2)
        k=k.view(intermim_shape).transpose(1,2)
        v=v.view(intermim_shape).transpose(1,2)
        
        # now we are applying the attention formula 
        # weight will be of the shape (batch_size,H,seq_len,seq_len)
        weight = q@k.transpose(-1,-2)
        
        if causal_mask:
            # mask where upper triangle is made of 1 
            mask = torch.ones_like(weight,dtype=torch.bool).triu(1)
            weight.masked_fill_(mask,-torch.inf)
        
        #  applying the second part of attention formula 
        weight /= mask.sqrt(self.d_head)
        weight = F.softmax(weight,dim=1)
        
        # (batch_size,h,seq_len,seq_len) @(batch_size,h,seq_len,dim/h) -> (batch_size,h,seq_len,dim/h)  
        output = weight @ v 
        
        # (batch_size,h,seq_len,dim/h) -> (batch_size,seq_len,h,dim/h) 
        output=output.transpose(1,2)
        
        # (batch_size,seq_len,h,dim/h)  ->(batch_size,seq_len,dim)
        output = output.reshape(input_shape)
        
        # mulitplying with W0 matrix 
        output = self.out_proj(output)
        
        return output
