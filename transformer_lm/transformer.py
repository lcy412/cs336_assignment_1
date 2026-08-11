from cs336_basics.src.transformer_lm import attention, functional, linear
import torch

def transformer_block(d_model: int, num_heads: int, d_ff: int,max_seq_len: int, theta: float, weights: torch.Tensor, in_features: torch.Tensor) -> torch.Tensor:
    
    q_proj_weight=weights["attn.q_proj.weight"]
    k_proj_weight=weights["attn.k_proj.weight"]
    v_proj_weight=weights["attn.v_proj.weight"]
    o_proj_weight=weights["attn.output_proj.weight"]
    g1=weights["ln1.weight"]
    g2=weights["ln2.weight"]
    w1=weights["ffn.w1.weight"]
    w2=weights["ffn.w2.weight"]
    w3=weights["ffn.w3.weight"]
    seq_len=in_features.shape[1]
    token_positions=torch.arange(0,seq_len)
    
    x=in_features
    

    rms_norm_1=functional.RMSNorm(d_model)
    rms_norm_1.gain.data=g1
    
    rms_norm_2=functional.RMSNorm(d_model)
    rms_norm_2.gain.data=g2
    
    pffn=functional.PositionWiseFFN(d_model)
    
    pffn.W1.weight.data=w1
    pffn.W2.weight.data=w2
    pffn.W3.weight.data=w3
    
    x_normed_1=rms_norm_1(x)
    
    
    x=x+attention.multihead_self_attention(d_model,num_heads,q_proj_weight,k_proj_weight,v_proj_weight,o_proj_weight,x_normed_1,token_positions,theta)
    
    x_normed_2=rms_norm_2(x)
    
    x=x+pffn(x_normed_2)
    
    return x
    
    
def transformer_lm(d_model: int, num_heads: int, d_ff: int, theta: float, weights: torch.Tensor, in_features: torch.Tensor, vocab_size: int, context_length: int,num_layers: int) -> torch.Tensor:
    embedding=linear.Embedding(vocab_size,d_model)
    embedding.weight.data=weights["token_embeddings.weight"]
    x=embedding(in_features)
    
    def extract_block_weights(state_dict: dict, layer_idx: int) -> dict:
        block_weights={}
        prefix=f"layers.{layer_idx}."
        
        for k,v in state_dict.items():
            if k.startswith(prefix):
                new_k=k[len(prefix):]
                block_weights[new_k]=v
                
        return block_weights
    
    for _ in range(num_layers):
        local_weights=extract_block_weights(weights,_)
        x=transformer_block(d_model, num_heads, d_ff,context_length, theta, local_weights, x)
        
    norm=functional.RMSNorm(d_model)
    norm.gain.data=weights["ln_final.weight"]
    x=norm(x)
    
    lm_head=linear.Linear(d_model,vocab_size)
    lm_head.weight.data=weights["lm_head.weight"]
    x=lm_head(x)
    
    return x