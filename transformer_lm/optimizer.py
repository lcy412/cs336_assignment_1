import math
from typing import Callable, Optional

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                lr = group["lr"]
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t"
                , 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
        state["t"] = t + 1 # Increment iteration number.
    
    
class AdamW(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1):
            if lr < 0:
                raise ValueError(f"Invalid learning rate: {lr}")
            if eps < 0:
                raise ValueError(f"Invalid epsilon value: {eps}")
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
            defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
            
            super().__init__(params, defaults)
            
            
        def step(self, closure: Optional[Callable] = None):
            loss = None if closure is None else closure()
            for group in self.param_groups:
                for p in group["params"]:
                    lr=group['lr']
                    betas=group['betas']
                    eps=group['eps']
                    weight_decay=group['weight_decay']
                    if p.grad is None:
                        continue
                    state = self.state[p] # Get state associated with p.
                    grad = p.grad.data 
                    t = state.get("t", 1)
                    m = state.get("m",torch.zeros_like(p.data))
                    v = state.get("v",torch.zeros_like(p.data))
                    lr_t=lr*math.sqrt(1-betas[1]**(t))/(1-betas[0]**(t))
                    
                    p.data -= lr*weight_decay*p.data
                    m = betas[0]*m + (1-betas[0])*grad
                    v = betas[1]*v + (1-betas[1])*grad*grad
                    
                    p.data -= lr_t*m/(torch.sqrt(v)+eps)
                    state["m"] = m
                    state["v"] = v
                    
                    state["t"] = t + 1
                    
                    
                    
                    
                    
                        
            
            
            
            
            
            
