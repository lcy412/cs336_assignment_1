from collections.abc import Iterable
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
    def __init__(self, d_model: int, d_ff:int |None=None,device=None, dtype=None):
        super().__init__()  
        if d_ff is not None:
            self.d_ff=d_ff
        else:
            self.d_ff=int(np.ceil(d_model*(8/3)/64))*64
        
        self.W1=Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2=Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.W3=Linear(d_model, self.d_ff, device=device, dtype=dtype)
        
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x*torch.sigmoid(x)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.silu(self.W1(x))*self.W3(x))
      
        
        
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_value=torch.max(x,dim=dim,keepdim=True)[0]
    
    return torch.exp(x-max_value)/torch.sum(torch.exp(x-max_value),dim=dim,keepdim=True)



def silu(x: torch.Tensor) -> torch.Tensor:
    return x*torch.sigmoid(x)
    
        
def cross_entropy(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    max_value=torch.max(x,dim=-1,keepdim=True)[0]
    x=x-max_value
    return torch.mean((torch.log(torch.sum(torch.exp(x),dim=-1,keepdim=True)))-(torch.gather(x, dim=-1, index=target.unsqueeze(-1))))
    
    
def get_lr_schedule(step: int, max_lr: float, min_lr: float, t_warmup : int, t_final:int) -> float:
    if step<t_warmup:
        return (step/t_warmup)*max_lr
    if step>=t_warmup and step<=t_final:
        return min_lr+0.5*(1+np.cos(np.pi*(step-t_warmup)/(t_final-t_warmup)))*(max_lr-min_lr)
    if step>t_final:
        return min_lr
    
def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float = 1.0) -> None:
    eps=1e-6
    l2_norm_sum=0
    params=list(params)
    for param in params:
        if param.grad is not None:
            l2_norm_sum+=torch.sum(param.grad.data**2).item()
    l2_norm_sum=np.sqrt(l2_norm_sum)
            
    if l2_norm_sum>max_norm:
        for param in params:
            if param.grad is not None:
                param.grad.data*=max_norm/(l2_norm_sum+eps)
                           
    return None
                
            
            
            
        
        
        
    
    
