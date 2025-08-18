from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    """
    Implements the Stochastic Gradient Descent algorithm with momentum.

    Arguments:
    """
    def __init__(self, params, lr = 1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                state["m"] = m
                state["v"] = v
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data += -alpha_t * m / (torch.sqrt(v) + epsilon)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
        return loss
