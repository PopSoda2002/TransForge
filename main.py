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
    for i in range(cfg.training.max_iterations):
        batch_data = get_batch_data(data, cfg.model.batch_size, cfg.model.context_length, cfg.model.device)
        x, y = next(batch_data)
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

    wandb.finish()


if __name__ == "__main__":
    train()