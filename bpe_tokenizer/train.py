from cs336_basics.src.bpe_tokenizer.pretokenizer import pretokenize
import collections
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

def merge(pretoken_counts: dict[tuple[bytes], int], pair:tuple[bytes, bytes]) -> dict[tuple[bytes], int] :
    
    
    merged_pretoken_counts={}
    
    for pretoken_tuple in pretoken_counts:
        i=0
        merged_pretoken_tuple=[]
        while i < len(pretoken_tuple):
            if i < len(pretoken_tuple)-1 and pretoken_tuple[i]==pair[0] and pretoken_tuple[i+1]==pair[1]:
                merged_pretoken_tuple.append(pair[0]+pair[1])
                i+=2
            else:
                merged_pretoken_tuple.append(pretoken_tuple[i])
                i+=1
        merged_pretoken_counts[tuple(merged_pretoken_tuple)]=pretoken_counts[pretoken_tuple]
    
    return merged_pretoken_counts
        
        
        
def get_pretoken_counts(pretoken_list: list[list[bytes]]) -> dict[tuple[bytes], int]:
    
    pretoken_counts={}
    for pretoken in pretoken_list:
        key=tuple(pretoken)
        pretoken_counts[key]=pretoken_counts.get(key,0)+1
    return pretoken_counts


def get_pair_counts_and_inverted_index(pretoken_counts: dict[tuple[bytes], int], special_token_tuples: tuple[tuple[bytes]]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    pairs_count={}
    inverted_index=collections.defaultdict(set)
    
    for pretoken_tuple, count in pretoken_counts.items():
        
        if pretoken_tuple in special_token_tuples:
            continue
        if len(pretoken_tuple)==1:
            continue
        for i in range(len(pretoken_tuple)-1):
            pair=(pretoken_tuple[i],pretoken_tuple[i+1])
            pairs_count[pair]=pairs_count.get(pair,0)+count
            
            
            inverted_index[pair].add(pretoken_tuple)
            

            
    if not pairs_count:
        return {},{}

    # pair = max(pairs_count,key=lambda x: (pairs_count[x],x))
    return pairs_count, inverted_index



           
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
            pretoken_counts=merge(pretoken_counts,pair)
            
            
    return vocab, merges
        
                
# if __name__ == "__main__":
#     input_file = "/Users/lichenyang/Documents/本科/大四/cs336_amnts/assignment1-basics/data/owt_valid.txt"
#     vocab_size = 1000
#     special_tokens = ["<|endoftext|>"]
#     train_bpe_tokenizer(input_file, vocab_size, special_tokens)
                
        
        
