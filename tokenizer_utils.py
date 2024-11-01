import tiktoken
import torch
from typing import List, Dict, Optional

class EnhancedTokenizer:
    def __init__(self):
        # 基础tokenizer
        self.base_tokenizer = tiktoken.get_encoding("gpt2")
        
        # 特殊token的起始ID (在GPT-2词表大小50257之后)
        self.special_tokens = {
            '<start>': 50257,  # 句子开始
            '<end>': 50258,    # 句子结束
            '<punct>': 50259,  # 标点符号
        }
        
        # 缓存常用token
        self.period_token = self.base_tokenizer.encode('.')[0]
        self.space_token = self.base_tokenizer.encode(' ')[0]
        
        # 词表大小 (基础词表 + 特殊token)
        self.vocab_size = 50257 + len(self.special_tokens)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """增强的编码函数"""
        if not add_special_tokens:
            return self.base_tokenizer.encode(text)
            
        # 分句处理
        sentences = text.split('.')
        encoded = []
        
        for sent in sentences:
            if sent.strip():
                # 基础token
                tokens = self.base_tokenizer.encode(sent.strip())
                # 添加特殊token
                if add_special_tokens:
                    tokens = [self.special_tokens['<start>']] + tokens + [self.special_tokens['<end>']]
                encoded.extend(tokens)
                
                # 如果不是最后一个句子，添加句号
                if sent != sentences[-1]:
                    encoded.append(self.period_token)
        
        return encoded
    
    def decode(self, tokens: List[int]) -> str:
        """解码函数"""
        # 分离特殊token和普通token
        normal_tokens = []
        for token in tokens:
            if token < 50257:  # 基础词表大小
                normal_tokens.append(token)
                
        # 解码普通token
        text = self.base_tokenizer.decode(normal_tokens)
        return text
    
    def get_vocab_size(self) -> int:
        """获取词表大小"""
        return self.vocab_size
    
    def is_special_token(self, token: int) -> bool:
        """判断是否是特殊token"""
        return token >= 50257
    
    def get_special_tokens_mask(self, tokens: List[int]) -> List[int]:
        """获取特殊token的mask"""
        return [1 if self.is_special_token(token) else 0 for token in tokens]

# 创建tokenizer实例的函数
def create_tokenizer() -> EnhancedTokenizer:
    return EnhancedTokenizer()