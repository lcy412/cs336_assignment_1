from cs336_basics.src.bpe_tokenizer.pretokenizer import pretokenize, split_by_special_tokens
import collections
from collections.abc import Iterator
from typing import overload
import os
from multiprocessing import Pool




class Node:
    def __init__(self, token: bytes):
        self.token = token
        self.prev: Node | None = None
        self.next: Node | None = None
        self.count: int = 0
        
class OrderedSet:
    
    def __init__(self):
        self._d = {}
    def add(self, x):
        self._d[x] = None
    def remove(self, x):
        del self._d[x]
    def discard(self, x):
        self._d.pop(x, None)
    def pop(self):
        k = next(iter(self._d))
        del self._d[k]
        return k
    def __bool__(self):
        return bool(self._d)
    def __contains__(self, x):
        return x in self._d     
    def clear(self):
        self._d.clear()   
    def __class_getitem__(cls, item):
        return cls
        

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
        
    

def merge(inverted_index: dict[tuple[bytes,bytes],OrderedSet[Node]], pair:tuple[bytes, bytes], pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes,bytes],OrderedSet[Node]]] :
    while inverted_index[pair]:
        node=inverted_index[pair].pop()
        pretoken_count=node.count
        
        if node.token==pair[0] and node.next is not None and node.next.token==pair[1]:
        
            new_token=pair[0]+pair[1]
            
            
            if node.next.next is not None:
                
                old_pair_tail=(pair[1],node.next.next.token)
                inverted_index[old_pair_tail].remove(node.next)
                
                node.next=node.next.next
                node.next.prev=node
                new_pair_tail=(new_token,node.next.token)
                inverted_index[new_pair_tail].add(node)
                pair_counts[new_pair_tail]=pair_counts.get(new_pair_tail,0)+pretoken_count
                pair_counts[old_pair_tail]=pair_counts.get(old_pair_tail,0)-pretoken_count
                # inverted_index[pair].remove(node)
                pair_counts[pair]=pair_counts.get(pair,0)-pretoken_count
            else:
                
                node.next=None
                pair_counts[pair]=pair_counts.get(pair,0)-pretoken_count
                
            if node.prev is not None:
                new_pair_head=(node.prev.token,new_token)
                old_pair_head=(node.prev.token,pair[0])
                inverted_index[new_pair_head].add(node.prev)
                inverted_index[old_pair_head].remove(node.prev)
                pair_counts[new_pair_head]=pair_counts.get(new_pair_head,0)+pretoken_count
                pair_counts[old_pair_head]=pair_counts.get(old_pair_head,0)-pretoken_count
            
            node.token=new_token
        else:
            continue


    
        
    inverted_index[pair].clear()  
    pair_counts[pair]=0
    
    return pair_counts, inverted_index
        
    
    

       
        


def initialize_state(pretoken_counts: dict[tuple[bytes],int]) -> tuple[dict[tuple[bytes], int],dict[Node,int],dict[tuple[bytes,bytes],OrderedSet[Node]]]:

    
    pretoken_chains={}
    pair_counts={}
    inverted_index=collections.defaultdict(OrderedSet)
    
    
    for pretoken_tuple,count in pretoken_counts.items():
        node=Node(pretoken_tuple[0])
        node.count=count
        pretoken_chains[node]=count
        for i in range(1,len(pretoken_tuple)):
            next_node=Node(pretoken_tuple[i])
            node.next=next_node
            next_node.prev=node
            next_node.count=count
            node=next_node
      
    for node, count in pretoken_chains.items():
        while node.next!=None:
            pair=(node.token,node.next.token)
            inverted_index[pair].add(node)
            pair_counts[pair]=pair_counts.get(pair,0)+count
            node=node.next
            
    
    
    return pair_counts, pretoken_chains, inverted_index

def get_pretoken_counts(pretoken_list: list[list[bytes]]) -> dict[tuple[bytes], int]:
    pretoken_counts={}
    for pretoken in pretoken_list:
        key=tuple(pretoken)
        pretoken_counts[key]=pretoken_counts.get(key,0)+1
    return pretoken_counts
           
def train_bpe_tokenizer(input_file:str, vocab_size:int, special_tokens: list[str]):
    num_workers = os.cpu_count()
    vocab=initialize_vocab(special_tokens)
    

    merges=[]
        
        
    with open(input_file, "r", encoding='utf-8') as f:
        text=f.read()      
        
        documents = split_by_special_tokens(text, special_tokens)
        chunk_size = max(1, len(documents) // num_workers)
    
        chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
        
        with Pool(num_workers) as pool:
            results = pool.starmap(process_chunk, [(chunk, special_tokens) for chunk in chunks])
            
            
        global_counts = {}
        for result in results:
            for key, count in result.items():
                global_counts[key] = global_counts.get(key, 0) + count
                
        pair_counts, pretoken_chains, inverted_index = initialize_state(global_counts)
        
        
        # pretokenized_tokens=pretokenize(text,special_tokens)
        # pretoken_list=list(transform_strs_to_bytes(pretokenized_tokens,special_tokens))
        
        # pretoken_counts=get_pretoken_counts(pretoken_list)
        # pair_counts,pretoken_chains,inverted_index=initialize_state(pretoken_counts)
        
        
        while len(vocab)<vocab_size:
            if not pair_counts:
                break
            pair=max(pair_counts,key=lambda x: (pair_counts[x],x))
            new_token=pair[0]+pair[1]
            vocab[len(vocab)]=new_token
            pair_counts, inverted_index=merge(inverted_index,pair,pair_counts)
            merges.append(pair)
            
            
    return vocab, merges


    

def process_chunk(document_chunks:list[str],special_tokens:list[str])->dict[tuple[bytes], int]:
    pretoken_counts={}
    for document in document_chunks:
        pretokenized_tokens=pretokenize(document,special_tokens)
        
        for pretoken in transform_strs_to_bytes(pretokenized_tokens, special_tokens):
            key=tuple(pretoken)
            pretoken_counts[key]=pretoken_counts.get(key,0)+1
            
    return pretoken_counts
    
    
    
        

                
        
        
