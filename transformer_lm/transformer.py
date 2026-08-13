from cs336_basics.src.transformer_lm import attention, functional, linear
import torch

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len:int,theta: float, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.d_ff=d_ff
        self.device=device
        self.dtype=dtype
        
        self.multihead_self_attention=attention.MultiHeadSelfAttention(d_model, num_heads, theta, device, dtype)
        self.positionwise_ffn=functional.PositionWiseFFN(d_model, d_ff, device, dtype)
        self.rms_norm_1=functional.RMSNorm(d_model)
        self.rms_norm_2=functional.RMSNorm(d_model)
        
    def forward(self,in_features: torch.Tensor,token_positions: torch.Tensor|None=None) -> torch.Tensor:
        
        x=in_features
        if token_positions is None:
            seq_len=in_features.shape[1]
            token_positions=torch.arange(seq_len).to(self.device, self.dtype)
        
        x_normed_1=self.rms_norm_1(x)
        
        x=x+self.multihead_self_attention.forward(x_normed_1,token_positions)
        
        x_normed_2=self.rms_norm_2(x)
        
        x=x+self.positionwise_ffn(x_normed_2)
        
        return x         
        
    
    
    
class TransformerLM(torch.nn.Module):
    def __init__(self,d_model: int, num_heads: int, d_ff: int, theta: float, vocab_size: int, context_length: int,num_layers: int, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.theta=theta
        self.max_seq_len=context_length
        self.d_ff=d_ff
        self.device=device
        self.dtype=dtype
        
        self.embedding=linear.Embedding(vocab_size,d_model)
        
        self.blocks=torch.nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device, dtype))
            
        self.norm=functional.RMSNorm(d_model)
        self.lm_head=linear.Linear(d_model,vocab_size)
        
    def forward(self,in_features: torch.Tensor,token_positions: torch.Tensor|None=None) -> torch.Tensor:
        if token_positions is None:
            seq_len=in_features.shape[1]
            token_positions=torch.arange(seq_len).to(self.device, self.dtype)
        
        x=self.embedding(in_features)
        
        for block in self.blocks:
            x=block(x,token_positions)
            
        x=self.norm(x)
        
        x=self.lm_head(x)
        
        return x
        
        


