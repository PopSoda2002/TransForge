import torch
from torch import Tensor


def get_next_token(logits: Tensor, temperature: float, top_p: float) -> int:
    # logits: [vocab_size]
    normed_logits = logits / temperature
    probs = torch.softmax(normed_logits, dim=-1)
    
    # sort in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    # create mask: retain tokens with cumulative probability <= top_p (but at least retain the first one)
    mask = cumulative_probs - sorted_probs <= top_p
    
    # only retain probabilities within the mask
    filtered_probs = sorted_probs * mask
    
    # re-normalize
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    token_index = torch.multinomial(filtered_probs, num_samples=1)
    next_token = sorted_indices[token_index]
    
    return next_token.item()

def generate_text(model, rope, tokenizer, prompt_tokens, max_length, temperature, top_p, eos_token_id, context_length, device):
    """自回归生成文本"""
    model.eval()
    
    # 确保 prompt_tokens 形状为 [1, seq_len]
    if prompt_tokens.dim() == 1:
        current_tokens = prompt_tokens.unsqueeze(0).to(device)
    else:
        current_tokens = prompt_tokens.to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(max_length):
            if i == max_length - 1:
                print("[Stop] Max length reached")
                break
            logits = model(current_tokens, rope)
            next_token_logits = logits[0, -1, :]
            next_token = get_next_token(next_token_logits, temperature, top_p)
            
            if eos_token_id is not None and next_token == eos_token_id:
                print("[Stop] EOS token encountered")
                break
            
            generated_tokens.append(next_token)
            
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            current_tokens = torch.cat([current_tokens, next_token_tensor], dim=1)
            
            # truncate context window
            if current_tokens.shape[1] > context_length:
                current_tokens = current_tokens[:, -context_length:]
    
    decoded_text = tokenizer.decode(generated_tokens)
    return decoded_text