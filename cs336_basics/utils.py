import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def stable_cross_entropy(logits, labels):
    """
    Compute numerically stable cross-entropy loss for classification.
    Args:
        logits: Tensor of shape (batch_size, num_classes)
        labels: Tensor of shape (batch_size,)
    Returns:
        Scalar tensor: mean cross-entropy loss over the batch
    """
    # For numerical stability, subtract the max logit from each row
    logits_max = torch.max(logits, dim=-1, keepdim=True).values  # shape: (batch_size, 1)
    logits_shifted = logits - logits_max  # shape: (batch_size, num_classes)
    exp_logits = torch.exp(logits_shifted)
    exp_logits_sum = exp_logits.sum(dim=-1)  # shape: (batch_size,)
    log_sum_exp = torch.log(exp_logits_sum)  # shape: (batch_size,)
    # Gather the logit corresponding to the correct label
    correct_class_logit = torch.gather(logits_shifted, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # Cross-entropy loss for each sample
    loss = log_sum_exp - correct_class_logit
    return loss.mean()

def perplexity_from_logits(logits, labels):
    """
    Compute perplexity for a given set of logits and labels.
    Args:
        logits: Tensor of shape (batch_size, num_classes)
        labels: Tensor of shape (batch_size,)
    Returns:
        Scalar tensor: perplexity over the batch
    """
    return torch.exp(stable_cross_entropy(logits, labels))

def perplexity_from_loss(loss):
    return torch.exp(loss)
    
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad == None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0, eps=1e-8):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad == None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                m = betas[0] * m + (1 - betas[0]) * p.grad.data
                v = betas[1] * v + (1 - betas[1]) * p.grad.data ** 2
                adjust = lr * (math.sqrt(1 - betas[1] ** (t + 1)) / (1 - betas[0] ** (t + 1)))
                p.data = p.data - adjust * m / (torch.sqrt(v) + eps) - lr * weight_decay * p.data
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
        return loss

def cosine_cycle_schedule(t, Tw, Tc, lr_min=1e-4, lr_max=1e-2):
    """
    Cosine cycle schedule for learning rate.
    """
    if t < Tw:
        return t / Tw * lr_max
    if t <= Tc:
        return lr_min + 0.5 * (1 + math.cos((t - Tw)/(Tc - Tw) * math.pi)) * (lr_max - lr_min)
    return lr_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    # Calculate the total L2 norm of all gradients
    total_norm = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    
    # If total norm exceeds max_l2_norm, scale all gradients
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.data.mul_(clip_coef)

if __name__ == "__main__":
    for lr in [1e1, 1e2, 1e3]:
        print(f"Learning rate: {lr}")
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        for t in range(10):
            opt.zero_grad() # Reset the gradients for all learnable parameters. 
            loss = (weights**2).mean() # Compute a scalar loss value. 
            print(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients. 
            opt.step() # Run optimizer step.