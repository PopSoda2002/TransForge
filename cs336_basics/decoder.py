import torch
from torch import Tensor


def get_next_token(logits: Tensor, temperature: float, top_p: float) -> int:
    # logits 应该是 [vocab_size] 形状的1D张量
    normed_logits = logits / temperature
    probs = torch.softmax(normed_logits, dim=-1)
    
    # 降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    # 创建mask：保留累积概率 <= top_p 的token（但至少保留第一个）
    mask = cumulative_probs - sorted_probs <= top_p
    
    # 只保留mask内的概率
    filtered_probs = sorted_probs * mask
    
    # 重新归一化
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    token_index = torch.multinomial(filtered_probs, num_samples=1)
    next_token = sorted_indices[token_index]
    
    return next_token.item()

def decode_with_top_p(x: Tensor, temperature: float, top_p: float, max_length: int, EOS_TOKEN_ID) -> str:
    for _ in range(max_length):
        next_token = get_next_token(x, temperature, top_p)
        if next_token == EOS_TOKEN_ID:
            break
        x = torch.cat([x, next_token], dim=0)
    return x