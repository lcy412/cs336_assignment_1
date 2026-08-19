import torch
from cs336_basics.src.transformer_lm.functional import softmax

def top_p_filter(probs, top_p):
    sorted_probs, sorted_indices=torch.sort(probs,dim=-1,descending=True)
    
    cumulative_probs=torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs - sorted_probs >=top_p
    sorted_probs[mask] = 0
    
    filtered_probs=torch.zeros_like(probs).scatter_(-1,sorted_indices,sorted_probs)
    
    filtered_probs=filtered_probs/torch.sum(filtered_probs,dim=-1,keepdim=True)
    
    return filtered_probs

    
    
def generate(input_ids,model, max_new_tokens, temperature=0.7, top_p=1, end_token_id=None,device=None):
    model.eval()
    input_ids=input_ids.to(device)
    
    if input_ids.dim() == 1:
        input_ids=input_ids.unsqueeze(0)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            output=model(input_ids)
            
        logits=output[:, -1, :]
        if temperature>0:
            logits=logits/temperature
        
        probs=softmax(logits,dim=-1)
        
        if top_p<1:
            probs=top_p_filter(probs, top_p)
            
            
        next_token_id=torch.multinomial(probs,num_samples=1)
        
        
        if end_token_id is not None and next_token_id.item() == end_token_id:
            break
        
        input_ids=torch.cat([input_ids, next_token_id], dim=-1)
        
    return input_ids
    
    
    
    
    
    
    
        
