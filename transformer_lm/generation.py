def generate(input_ids,model, max_new_tokens, temperature=0.7, top_p=1, end_token_id='None',device=None):
    logits=model(input_ids)
    logits=logits[:, -1, :]
    
    
    
    
        
