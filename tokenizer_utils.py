import tiktoken
import torch
from typing import List, Dict, Optional

class EnhancedTokenizer:
    def __init__(self):
        self.base_tokenizer = tiktoken.get_encoding("gpt2")

        self.special_tokens = {
            '<start>': 50257, 
            '<end>': 50258,    
            '<punct>': 50259,  
        }

        self.period_token = self.base_tokenizer.encode('.')[0]
        self.space_token = self.base_tokenizer.encode(' ')[0]

        self.vocab_size = 50257 + len(self.special_tokens)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if not add_special_tokens:
            return self.base_tokenizer.encode(text)

        sentences = text.split('.')
        encoded = []
        
        for sent in sentences:
            if sent.strip():
                tokens = self.base_tokenizer.encode(sent.strip())
                if add_special_tokens:
                    tokens = [self.special_tokens['<start>']] + tokens + [self.special_tokens['<end>']]
                encoded.extend(tokens)

                if sent != sentences[-1]:
                    encoded.append(self.period_token)
        
        return encoded
    
    def decode(self, tokens: List[int]) -> str:

        normal_tokens = []
        for token in tokens:
            if token < 50257:  
                normal_tokens.append(token)

        text = self.base_tokenizer.decode(normal_tokens)
        return text
    
    def get_vocab_size(self) -> int:

        return self.vocab_size
    
    def is_special_token(self, token: int) -> bool:

        return token >= 50257
    
    def get_special_tokens_mask(self, tokens: List[int]) -> List[int]:

        return [1 if self.is_special_token(token) else 0 for token in tokens]

def create_tokenizer() -> EnhancedTokenizer:
    return EnhancedTokenizer()
