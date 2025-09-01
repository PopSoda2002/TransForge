import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[Tensor, "batch_size context_length vocab_size"],
    targets: Int[Tensor, "batch_size context_length"],
) -> Float[Tensor, ""]:
    # Flatten the inputs and targets to treat the sequence dimension as part of the batch
    inputs_flat = inputs.view(-1, inputs.shape[-1])
    targets_flat = targets.view(-1)

    max_logit = inputs_flat.max(dim=-1, keepdim=True).values
    logit_stable = inputs_flat - max_logit
    
    log_sum_exp = torch.log(torch.sum(torch.exp(logit_stable), dim=-1, keepdim=False))
    
    # Select the logits for the target classes from the flattened tensors
    target_logits = logit_stable[torch.arange(inputs_flat.shape[0]), targets_flat]
    
    log_probs = target_logits - log_sum_exp
    
    return -torch.mean(log_probs)