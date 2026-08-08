import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for e in range(epochs):
            torch.manual_seed(e)
            idx_start_list = torch.randint(
                len(data) - context_length,
                (batch_size,)
            )
            
            X_list, y_list = [], []
            for i in idx_start_list:
                X_list.append(data[i : i + context_length])
                y_list.append(data[i + 1 : i + context_length + 1])
            
            X = torch.stack(X_list)
            y = torch.stack(y_list)

            logits = model(X)
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            y = y.reshape(B * T)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)