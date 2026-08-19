import torch
from einops import einsum,rearrange
import numpy as np
from cs336_basics.src.transformer_lm.functional import softmax
from cs336_basics.src.transformer_lm.linear import Linear
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
        

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k=Q.shape[-1]
    
    if mask is not None:
        if mask.device != Q.device:
            mask = mask.to(Q.device)
        num_mask=torch.where(mask,0,-np.inf)
        
        attention_scores=(einsum(Q,K,"... queries d_k, ... keys d_k -> ... queries keys"))/np.sqrt(d_k)+num_mask
        return softmax(attention_scores)@V
    else:
        return softmax((einsum(Q,K,"... queries d_k, ... keys d_k -> ... queries keys"))/np.sqrt(d_k))@V
    


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None=None, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.q_proj=Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj =Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj=Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj=Linear(d_model, d_model, device=device, dtype=dtype)
        self.theta=theta
        self.device=device
    
    def forward(
        self,
        in_features:torch.Tensor,
        token_positions: torch.Tensor|None=None,
    ) -> torch.Tensor:
        num_heads=self.num_heads
        d_model=self.d_model
        theta=self.theta
        
        seq_len=in_features.shape[-2]
        
        K=einsum(self.k_proj.weight,in_features,"... dk d_in, ... seq_len d_in -> ... seq_len dk")
        K=rearrange(K,"... seq_len (num_heads dk_head) -> ...  num_heads seq_len dk_head",num_heads=num_heads)
        
        Q=einsum(self.q_proj.weight,in_features,"... dk d_in, ... seq_len d_in -> ... seq_len dk")
        Q=rearrange(Q,"... seq_len (num_heads dk_head) -> ...  num_heads seq_len dk_head",num_heads=num_heads)
        
        if token_positions is not None:
            rpe=RotaryPositionalEmbedding(theta,d_model/num_heads,seq_len,self.device)
            Q=rpe(Q,token_positions)
            K=rpe(K,token_positions)
        
            
        
        V=einsum(self.v_proj.weight,in_features,"... dv d_in, ... seq_len d_in -> ... seq_len dv")
        V=rearrange(V,"... seq_len (num_heads dv_head) -> ...  num_heads seq_len dv_head",num_heads=num_heads)
        
        causal_mask=torch.tril(torch.ones((seq_len,seq_len),dtype=torch.bool))
        
        attention_scores=scaled_dot_product_attention(Q,K,V,causal_mask)
        attention_scores=rearrange(attention_scores,"... num_heads seq_len dv_head -> ... seq_len (num_heads dv_head)")
        
        multihead_attention=einsum(self.o_proj.weight,attention_scores,"... d_model dv, ... seq_len dv -> ... seq_len d_model")
        

        return multihead_attention
    
    
    
    
    
    
    
    
    
    
    



