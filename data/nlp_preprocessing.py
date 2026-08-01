import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_data = positive + negative
        vocab = set()
        for sent in all_data:
            for w in sent.split():
                vocab.add(w)
        vocab = sorted(vocab)
        word_to_id = {
            word: idx + 1 for idx, word in enumerate(vocab)
        }

        all_encoded = []
        for sent in all_data:
            encoded_sent = [word_to_id[w] for w in sent.split()]
            all_encoded.append(torch.tensor(encoded_sent))

        padded_all_encoded = nn.utils.rnn.pad_sequence(
            all_encoded, 
            batch_first=True,
            padding_value=0,
        )

        return padded_all_encoded
