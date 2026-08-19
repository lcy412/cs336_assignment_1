import json
import time
import numpy as np
import torch
import cs336_basics.src.transformer_lm.transformer as transformer
import cs336_basics.src.transformer_lm.optimizer as optimizer
import cs336_basics.src.transformer_lm.functional as functional
def get_batch(dataset: np.ndarray, batch_size: int, context_length:int, device: str ):
    
    off_tensor=torch.arange(0, context_length)
    off_tensor=off_tensor.reshape(1,-1)
    
    start_tensor=torch.randint(0,len(dataset)-context_length,(batch_size,))
    start_tensor=start_tensor.reshape(-1,1)
    
    index_tensor=start_tensor+off_tensor
    
    index_tensor=index_tensor
    
    input_data_tensor=torch.tensor(dataset[index_tensor],device=device) 
    target_data_tensor_tensor=torch.tensor(dataset[index_tensor+1],device=device) 
    
    
    return (input_data_tensor, target_data_tensor_tensor)   

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint={}
    if hasattr(model, "_orig_mod"):
        checkpoint["model"] = model._orig_mod.state_dict()
    else:
        checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"]=optimizer.state_dict()
    checkpoint["iteration"]=iteration
    
    torch.save(checkpoint, out)
    
def load_checkpoint(src, model, optimizer):
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration=checkpoint["iteration"]
    
    return iteration

def train(model, optimizer, dataset, batch_size, context_length, num_iterations, max_grad_norm, device, checkpoint_path,max_lr,min_lr,t_warmup,t_final,log_path,eval_interval=None, checkpoint_interval=None, log_interval=None,seed=None):
    model.train()
    model.to(device)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
         
    start_time=time.time()
    for iteration in range(num_iterations):
        
        
        inputs, targets=get_batch(dataset, batch_size, context_length, device)
        
        logits=model(inputs)
        
        loss=functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        functional.gradient_clipping(model.parameters(), max_grad_norm)
        lr=functional.get_lr_schedule(iteration,max_lr,min_lr,t_warmup,t_final)
        for group in optimizer.param_groups:
            group['lr']=lr
        optimizer.step()
        
        if log_interval is not None and iteration % log_interval == 0:
            elapsed=time.time()-start_time
            log_line = {
                "iteration": iteration,
                "loss": loss.item(),
                "lr": lr,
                "elapsed_time": elapsed
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_line) + "\n")
              
        if checkpoint_path is not None and checkpoint_interval is not None and iteration % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            
    if checkpoint_path is not None:
        save_checkpoint(model, optimizer, num_iterations - 1, checkpoint_path)

    
            
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
