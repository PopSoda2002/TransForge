import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import wandb

from cs336_basics.data_loader import get_batch_data
from cs336_basics.transformer import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.rope import RoPE
from cs336_basics.utils import learning_rate_cosine_scheduler, gradient_clipping
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.decoder import get_next_token

def generate_text(model, rope, tokenizer, prompt_tokens, max_length, temperature, top_p, eos_token_id, device, context_length):
    """自回归生成文本"""
    model.eval()
    
    # 确保 prompt_tokens 形状为 [1, seq_len]
    if prompt_tokens.dim() == 1:
        current_tokens = prompt_tokens.unsqueeze(0).to(device)
    else:
        current_tokens = prompt_tokens.to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # 获取模型输出: [1, seq_len, vocab_size]
            logits = model(current_tokens, rope)
            
            # 取最后一个位置的 logits: [vocab_size]
            next_token_logits = logits[0, -1, :]
            
            # 采样下一个 token
            next_token = get_next_token(next_token_logits, temperature, top_p)
            
            # 检查是否遇到 EOS token
            if eos_token_id is not None and next_token == eos_token_id:
                break
            
            generated_tokens.append(next_token)
            
            # 将新 token 添加到序列中
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            current_tokens = torch.cat([current_tokens, next_token_tensor], dim=1)
            
            # 如果超过 context_length，只保留最后 context_length 个 token
            if current_tokens.shape[1] > context_length:
                current_tokens = current_tokens[:, -context_length:]
    
    # 解码生成的 tokens
    decoded_text = tokenizer.decode(generated_tokens)
    return decoded_text


@hydra.main(config_path="conf", config_name="config.yaml")
def train(cfg: DictConfig):
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    rope = RoPE(cfg.model.rope_theta, cfg.model.d_model // cfg.model.num_heads, cfg.model.context_length).to(cfg.model.device)
    transformer = TransformerLM(cfg.model.vocab_size, cfg.model.num_layers, cfg.model.d_model, cfg.model.num_heads, cfg.model.d_ff)
    transformer.to(cfg.model.device)

    data = np.memmap(cfg.data.tokenized_path, dtype='uint16', mode='r')
    optimizer = AdamW(
        transformer.parameters(),
        lr=cfg.scheduler.lr_max,
        betas=(cfg.optimizer.betas[0], cfg.optimizer.betas[1]),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )
    train_data = get_batch_data(data[:int(0.9 * len(data))], cfg.model.batch_size, cfg.model.context_length, cfg.model.device)
    val_data = get_batch_data(data[int(0.9 * len(data)):], cfg.model.batch_size, cfg.model.context_length, cfg.model.device)
    bpe_tokenizer = BPETokenizer.from_files(cfg.tokenizer.vocab_path, cfg.tokenizer.merges_path, cfg.tokenizer.special_tokens)
    
    # 获取 EOS token ID
    eos_token_id = bpe_tokenizer.encode("<|endoftext|>")[0] if "<|endoftext|>" in bpe_tokenizer.special_tokens else None
    
    for i in range(cfg.training.max_iterations):
        x, y = next(train_data)
        x = x.to(device=cfg.model.device, dtype=torch.long)
        y = y.to(device=cfg.model.device, dtype=torch.long)
        optimizer.zero_grad()
        output = transformer(x, rope)
        loss = cross_entropy(output, y)
        loss.backward()
        gradient_clipping(transformer.parameters(), cfg.training.max_l2_norm)
        optimizer.step()
        
        # Log metrics to wandb
        current_lr = learning_rate_cosine_scheduler(i, cfg.scheduler.T_w, cfg.scheduler.T_c, cfg.scheduler.lr_max, cfg.scheduler.lr_min)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        wandb.log({"loss": loss.item(), "iteration": i, "lr": current_lr})
        if i % cfg.training.log_interval == 0:
            print(f"Iteration {i} loss: {loss.item()}")
            val_x, val_y = next(val_data)
            val_x = val_x.to(device=cfg.model.device, dtype=torch.long)
            val_y = val_y.to(device=cfg.model.device, dtype=torch.long)
            
            # 生成文本
            prompt_tokens = val_x[0]  # 取第一个样本作为 prompt
            decoded_text = generate_text(
                model=transformer,
                rope=rope,
                tokenizer=bpe_tokenizer,
                prompt_tokens=prompt_tokens,
                max_length=cfg.decoder.max_length,
                temperature=cfg.decoder.temperature,
                top_p=cfg.decoder.p,
                eos_token_id=eos_token_id,
                device=cfg.model.device,
                context_length=cfg.model.context_length
            )
            print(f"Iteration {i} decoded_text: {decoded_text}")
            
            val_loss = cross_entropy(transformer(val_x, rope), val_y)
            wandb.log({"val_loss": val_loss.item()})
            print(f"Iteration {i} val_loss: {val_loss.item()}")

    wandb.finish()


if __name__ == "__main__":
    train()