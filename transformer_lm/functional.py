import torch 
import numpy as np
from cs336_basics.src.transformer_lm.linear import Linear
from einops import einsum
from einops import rearrange

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.gain=torch.nn.Parameter(torch.randn(d_model, dtype=dtype, device=device))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms=torch.sqrt(torch.mean(torch.square(x),dim=-1,keepdim=True)+self.eps)
        
        result=(x*self.gain)/rms
        # Return the result in the original dtype
        return result.to(in_dtype)
    
class PositionWiseFFN(torch.nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()  
        self.d_ff=int(np.ceil(d_model*(8/3)/64))*64
        self.W1=Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2=Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.W3=Linear(d_model, self.d_ff, device=device, dtype=dtype)
        
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x*torch.sigmoid(x)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.silu(self.W1(x))*self.W3(x))
    
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.device=device
        frequency_vec=torch.arange(0,d_k,2,device=device)/d_k
        frequency_vec=self.theta**(-frequency_vec)
        self.frequency_vec=frequency_vec
        
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        self.angle=einsum(token_positions,self.frequency_vec,"... seq_len, f-> ... seq_len f ")
        cos_vec=torch.cos(self.angle)
        cos_vec=torch.repeat_interleave(cos_vec,2,dim=-1)
        sin_vec=torch.sin(self.angle)
        sin_vec=torch.repeat_interleave(sin_vec,2,dim=-1)
        
        x_even=x[...,0::2]
        x_odd=-x[...,1::2]
        
        x_flip=rearrange([x_odd,x_even], "two ... dk_2 -> ... ( dk_2 two)")
        
        return einsum(cos_vec,x,"... seq_len d_model,... seq_len d_model -> ... seq_len d_model")+einsum(sin_vec,x_flip,"... seq_len d_model,... seq_len d_model -> ... seq_len d_model")
        
        
        
        
        
        
        
        
        
        
        
        
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_value=torch.max(x,dim=dim,keepdim=True)[0]
    
    return torch.exp(x-max_value)/torch.sum(torch.exp(x-max_value),dim=dim,keepdim=True)


   
        

