import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    max_logit = inputs.max(dim=-1, keepdim=True).values
    logit_stable = inputs - max_logit
    
    log_sum_exp = torch.log(torch.sum(torch.exp(logit_stable), dim=-1, keepdim=False))
    
    target_logits = logit_stable[torch.arange(inputs.shape[0]), targets]
    
    log_probs = target_logits - log_sum_exp
    
    return -torch.mean(log_probs)