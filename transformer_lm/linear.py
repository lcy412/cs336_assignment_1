import torch 
import numpy as np

class Linear(torch.nn.Module):
    
    def __init__(self, in_features:int , out_features:int, device=None, dtype=None):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.randn(out_features, in_features, dtype=dtype, device=device))
        sigma=np.sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma,a=-3*sigma,b=3*sigma)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    
class Embedding(torch.nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1,a=-3,b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
    
        
        
        
        
        
