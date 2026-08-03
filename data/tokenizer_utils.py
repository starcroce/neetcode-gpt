from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        res = []
        for n in numbers:
            n = str(n)
            tokens = self._greedy_tokenize(n, vocab)
            res.append(tokens)
        return res

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens = self._greedy_tokenize(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens = self._greedy_tokenize(text, vocab)
        words = text.split()
        res = len(tokens) / len(words)
        return round(res, 4)

    def _greedy_tokenize(self, text, vocab):
        tokens = []
        i = 0
        while i < len(text):
            longest = None            
            for j in range(len(text), i, -1):
                if text[i:j] in vocab:
                    longest = text[i:j]
                    break
            if longest:
                tokens.append(longest)
                i = j
            else:
                tokens.append(text[i])
                i += 1
        return tokens

