import base64
import json
from typing import Iterable, Iterator
from cs336_basics.src.bpe_tokenizer.pretokenizer import pretokenize, split_by_special_tokens
from cs336_basics.src.bpe_tokenizer.train import transform_strs_to_bytes, get_pretoken_counts, initialize_state

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inverted_vocab = {v:k for k,v in self.vocab.items()}
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            vocab = {int(k): base64.b64decode(v) for k, v in vocab.items()}
            
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = []
            for line in f:
                encoded_a, encoded_b = line.strip().split()
                a = base64.b64decode(encoded_a)
                b = base64.b64decode(encoded_b)
                merges.append((a, b))
                
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        special_tokens = self.special_tokens if self.special_tokens is not None else []

        pretoken_strs=pretokenize(text,special_tokens)
        pretoken_list=list(transform_strs_to_bytes(pretoken_strs,special_tokens))
        
        new_list=[]
        for pretoken in pretoken_list:
            if len(pretoken)==1 and pretoken[0].decode('utf-8') in special_tokens:
                new_list.append(pretoken[0])
                continue
            
            for pair in self.merges:
                new_seq=[]
                i=0
                while i<len(pretoken):
                    if i<len(pretoken)-1 and (pretoken[i],pretoken[i+1])==pair:
                        new_seq.append(pair[0]+pair[1])
                        i+=2
                    else:
                        new_seq.append(pretoken[i])
                        i+=1
                pretoken=new_seq
                    
            new_list.extend(new_seq)
            
        
        return [self.inverted_vocab[token] for token in new_list]
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            for token_id in self.encode(text_chunk):
                yield token_id

                        
    def decode(self, ids: list[int]) -> str:
        all_bytes = b"".join(self.vocab[id] for id in ids)
        return all_bytes.decode('utf-8',errors='replace')
        
                
            
        
        
        
