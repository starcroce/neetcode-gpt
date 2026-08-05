import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        idx_start_list = torch.randint(
            0, 
            len(tokens) - context_length,
            (batch_size,),
        ).tolist()

        X, y = [], []
        for i in idx_start_list:
            X.append(tokens[i : i+context_length])
            y.append(tokens[i+1 : i+context_length+1])

        return X, y
