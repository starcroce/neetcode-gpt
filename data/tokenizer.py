from typing import List
from collections import defaultdict

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        res = []

        for _ in range(num_merges):
            pair_cnt = defaultdict(int)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_cnt[pair] += 1

            max_pair_cnt = max(pair_cnt.values())
            cands = sorted([p for p in pair_cnt if pair_cnt[p] == max_pair_cnt])
            res.append((cands[0][0], cands[0][1]))

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == cands[0]:
                    new_tokens.append("".join(cands[0]))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return res