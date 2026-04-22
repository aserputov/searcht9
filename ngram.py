"""
N-gram Language Model for Next-Token Prediction

Builds bigram and trigram frequency tables from a text corpus,
then predicts the most probable next tokens given a context.
This demonstrates the core statistical concept behind autoregressive
language models: P(next_token | previous_tokens).
"""

import re
from collections import defaultdict
import heapq


class NGramModel:
    """Statistical next-token prediction using n-gram frequency analysis."""

    def __init__(self, n=3):
        self.n = n  # max n-gram size (trigram by default)
        self.unigrams = defaultdict(int)
        self.bigrams = defaultdict(lambda: defaultdict(int))
        self.trigrams = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        self.total_tokens = 0

    def tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower().strip()
        tokens = re.findall(r"\b\w+\b|[.,!?;:]", text)
        return tokens

    def train(self, corpus: str):
        """Build n-gram frequency tables from text corpus."""
        sentences = re.split(r"[.!?]+", corpus)
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            if len(tokens) < 2:
                continue

            for i, token in enumerate(tokens):
                self.unigrams[token] += 1
                self.vocab.add(token)
                self.total_tokens += 1

                # Bigram: P(token | previous)
                if i >= 1:
                    prev = tokens[i - 1]
                    self.bigrams[prev][token] += 1

                # Trigram: P(token | prev2, prev1)
                if i >= 2:
                    context = (tokens[i - 2], tokens[i - 1])
                    self.trigrams[context][token] += 1

    def predict_next(self, context: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Predict most probable next tokens given context.

        Uses trigram if available, backs off to bigram, then unigram.
        Returns list of (token, probability) tuples.
        """
        tokens = self.tokenize(context)
        candidates = {}

        # Try trigram first (most specific context)
        if len(tokens) >= 2:
            ctx = (tokens[-2], tokens[-1])
            if ctx in self.trigrams:
                total = sum(self.trigrams[ctx].values())
                for token, count in self.trigrams[ctx].items():
                    candidates[token] = count / total

        # Back off to bigram
        if not candidates and len(tokens) >= 1:
            prev = tokens[-1]
            if prev in self.bigrams:
                total = sum(self.bigrams[prev].values())
                for token, count in self.bigrams[prev].items():
                    candidates[token] = count / total

        # Back off to unigram (least specific)
        if not candidates:
            for token, count in self.unigrams.items():
                candidates[token] = count / self.total_tokens

        # Return top-k by probability using heap
        top = heapq.nlargest(top_k, candidates.items(), key=lambda x: x[1])
        return top

    def perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text under this model.
        Lower perplexity = better prediction. This is the standard
        metric for evaluating language models.
        """
        tokens = self.tokenize(text)
        if len(tokens) < 2:
            return float("inf")

        log_prob_sum = 0.0
        n_predictions = 0

        import math

        for i in range(1, len(tokens)):
            token = tokens[i]

            # Try trigram
            prob = None
            if i >= 2:
                ctx = (tokens[i - 2], tokens[i - 1])
                if ctx in self.trigrams:
                    total = sum(self.trigrams[ctx].values())
                    count = self.trigrams[ctx].get(token, 0)
                    if count > 0:
                        prob = count / total

            # Back off to bigram
            if prob is None:
                prev = tokens[i - 1]
                if prev in self.bigrams:
                    total = sum(self.bigrams[prev].values())
                    count = self.bigrams[prev].get(token, 0)
                    if count > 0:
                        prob = count / total

            # Back off to unigram with smoothing
            if prob is None:
                prob = (self.unigrams.get(token, 0) + 1) / (
                    self.total_tokens + len(self.vocab)
                )

            log_prob_sum += math.log2(prob)
            n_predictions += 1

        if n_predictions == 0:
            return float("inf")

        return 2 ** (-log_prob_sum / n_predictions)

    def stats(self) -> dict:
        """Return model statistics."""
        return {
            "vocab_size": len(self.vocab),
            "total_tokens": self.total_tokens,
            "unique_bigrams": sum(len(v) for v in self.bigrams.values()),
            "unique_trigrams": sum(len(v) for v in self.trigrams.values()),
        }
