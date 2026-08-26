import math
import re
from collections import Counter
from itertools import pairwise

token_pattern = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}")


class LocalEmbeddingService:
    model_name = "local-hashing-v1"

    def embed(self, text: str) -> Counter:
        tokens = [token.lower() for token in token_pattern.findall(text)]
        grams = Counter(tokens)
        grams.update(" ".join(pair) for pair in pairwise(tokens))
        return grams

    def cosine(self, left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0
        intersection = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in intersection)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def keyword_overlap(self, question: str, text: str) -> float:
        q_terms = set(token.lower() for token in token_pattern.findall(question))
        t_terms = set(token.lower() for token in token_pattern.findall(text))
        if not q_terms:
            return 0.0
        return len(q_terms & t_terms) / len(q_terms)
