import regex as re
def split_by_special_tokens(tokens: str, special_tokens: list[str]) -> list[str]:
    """
    Split the input tokens by the special tokens.
    """
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    
    special_token_pattern="|".join(re.escape(special_token) for special_token in sorted_special_tokens)
    
    split_tokens=re.split(f"({special_token_pattern})", tokens)
    
    return [token for token in split_tokens if token != '']
    
def pretokenize_text(text: str,)-> list[str]:
    """
    Pre-tokenize the input text into a list of tokens.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    pretokens=[match.group() for match in re.finditer(PAT,text)]
    
    return pretokens

def pretokenize(tokens: str, special_tokens: list[str]) -> list[str]:
    text_list=split_by_special_tokens(tokens,special_tokens)
    result=[]
    for text in text_list:
        if text in special_tokens:
            result.append(text)
        else:
            result.extend(pretokenize_text(text))
    return result
        