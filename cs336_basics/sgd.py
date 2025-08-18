from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    """
    Implements the Stochastic Gradient Descent algorithm with momentum.

    Arguments:
    """
    def __init__(self, params, lr = 1e-3):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = {"lr": lr}
        super(SGD, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data += -lr * grad / math.sqrt(t + 1)
                state["t"] = t + 1

        return loss

weights = torch.nn.Parameter(5 * torch.randn(10, 10))
opt = SGD([weights], lr=0.01)

for t in range(100):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()
