import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        res = []
        with torch.no_grad():
            for m in model.children():
                x = m(x)
                if isinstance(m, nn.Linear):
                    mean = round(x.mean().item(), 4)
                    std = round(x.std().item(), 4)
                    if x.dim() >= 2:
                        dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    else:
                        dead_frac = round((x <= 0).float().mean().item(), 4)
                    res.append({
                        "mean": mean,
                        "std": std,
                        "dead_fraction": dead_frac,
                    })
        return res

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        outputs = model(x)
        loss = nn.MSELoss()(outputs, y)
        loss.backward()
        res = []

        for m in model.children():
            if isinstance(m, nn.Linear):
                grad = m.weight.grad
                mean = round(grad.mean().item(), 4)
                std = round(grad.std().item(), 4)
                norm = round(torch.norm(grad).item(), 4)
                res.append({
                    "mean": mean,
                    "std": std,
                    "norm": norm,
                })

        return res

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for s in activation_stats:
            if s["dead_fraction"] > 0.5:
                return "dead_neurons"
            if s["std"] < 0.1:
                return "vanishing_gradients"
            if s["std"] > 10:
                return "exploding_gradients"
        
        for s in gradient_stats:
            if s["norm"] > 1000:
                return "exploding_gradients"
            
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        return "healthy"