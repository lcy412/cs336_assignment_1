from cs336_basics.src.bpe_tokenizer.pretokenizer import pretokenize
from collections.abc import Iterator

def transform_strs_to_bytes(pretokens: list[str], special_tokens: list[str]) -> Iterator[list[bytes]]:
    
    for pretoken in pretokens:
        if pretoken in special_tokens:
            yield [pretoken.encode("utf-8")]
        else:
            pretokens_bytes = pretoken.encode("utf-8")
            yield[bytes([b]) for b in pretokens_bytes]
            
def initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    
    vocab={}
    for i in range(256):
        vocab[i]=bytes([i])
    for i in range(len(special_tokens)):
        vocab[256+i]=special_tokens[i].encode("utf-8")
        
    return vocab
        
def get_stats(pretoken_counts: dict[tuple[bytes], int], special_token_tuples: tuple[tuple[bytes]]) -> tuple[dict[tuple[bytes, bytes], int], tuple[bytes, bytes], None]:
    
    pairs_count={}
    
    for pretoken_tuple, count in pretoken_counts.items():
        
        if pretoken_tuple in special_token_tuples:
            continue
        if len(pretoken_tuple)==1:
            continue
        for i in range(len(pretoken_tuple)-1):
            pair=(pretoken_tuple[i],pretoken_tuple[i+1])
            pairs_count[pair]=pairs_count.get(pair,0)+count
            
    if not pairs_count:
        return {}, None

    pair = max(pairs_count,key=lambda x: (pairs_count[x],x))
    # pair = min(pairs_count, key=lambda x: (-pairs_count[x], x))


    return pairs_count, pair  

def merge(pretoken_list: list[list[bytes]], pair:tuple[bytes, bytes]) -> list[list[bytes]] :
    
    
    merged_pretoken_list=[]
    
    for pretoken in pretoken_list:
        new_token=[]
        i=0
        while i<len(pretoken):
            if i<len(pretoken)-1 and pretoken[i]==pair[0] and pretoken[i+1]==pair[1]:
                new_token.append(pretoken[i]+pretoken[i+1])
                i+=2
            else:
                new_token.append(pretoken[i])
                i+=1
        merged_pretoken_list.append(new_token)
        
    return merged_pretoken_list
    
def get_pretoken_counts(pretoken_list: list[list[bytes]]) -> dict[tuple[bytes], int]:
    
    pretoken_counts={}
    for pretoken in pretoken_list:
        key=tuple(pretoken)
        pretoken_counts[key]=pretoken_counts.get(key,0)+1
    return pretoken_counts
               
def train_bpe_tokenizer(input_file:str, vocab_size:int, special_tokens: list[str]):
    
    vocab=initialize_vocab(special_tokens)
    
    special_token_tuples = ((token.encode("utf-8"),) for token in special_tokens)

    merges=[]
        
        
    with open(input_file, "r", encoding='utf-8') as f:
        text=f.read()      
        pretokenized_tokens=pretokenize(text,special_tokens)
        pretoken_list=list(transform_strs_to_bytes(pretokenized_tokens,special_tokens))
        
        pretoken_counts=get_pretoken_counts(pretoken_list)
        
        
        while len(vocab)<vocab_size:
            
            pairs_count, pair=get_stats(pretoken_counts,special_token_tuples)
            if pair is None:
                break
            vocab[len(vocab)]=pair[0]+pair[1]
            merges.append(pair)
            pretoken_list=merge(pretoken_list,pair)
            pretoken_counts=get_pretoken_counts(pretoken_list)
            
    return vocab, merges
        
                
# if __name__ == "__main__":
#     input_file = "/Users/lichenyang/Documents/本科/大四/cs336_amnts/assignment1-basics/data/owt_valid.txt"
#     vocab_size = 1000
#     special_tokens = ["<|endoftext|>"]
#     train_bpe_tokenizer(input_file, vocab_size, special_tokens)
                
        
        
