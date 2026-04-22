# Statistical Next-Token Prediction Engine

A from-scratch implementation of frequency-based language modeling, demonstrating the core statistical concepts that underlie modern autoregressive LLMs.

## How It Works

Modern LLMs (GPT, Claude, Llama) predict the next token given previous context — `P(next_token | context)`. This project implements the same concept using classical n-gram statistics, showing the progression from simple frequency counting to context-aware prediction.

### Two prediction approaches:

**1. Prefix-Tree Autocomplete (Trie + Heap)**
- Character-level prefix matching with frequency-based ranking
- O(log n) top-k retrieval using a min-heap
- Learns from user queries in real-time

**2. N-gram Language Model (Bigram/Trigram)**
- Word-level next-token prediction: `P(word | prev_word)` (bigram), `P(word | prev2, prev1)` (trigram)
- Backoff strategy: trigram → bigram → unigram (if no context match found)
- Perplexity evaluation (standard LM metric — lower = better predictions)
- Trained on an English text corpus covering ML/systems/NLP domains

## Example

```
GET /predict?q=the model

{
  "context": "the model",
  "predictions": [
    {"token": "was", "probability": 0.25},
    {"token": "uses", "probability": 0.17},
    {"token": "weights", "probability": 0.12},
    {"token": "generates", "probability": 0.10},
    {"token": "achieved", "probability": 0.08}
  ]
}
```

The model predicts "was" as the most likely next word after "the model" because the training corpus contains phrases like "the model was trained...", "the model was fine tuned...".

## Connection to Modern LLMs

| Concept | This Project | Modern LLMs (GPT/Claude) |
|---------|-------------|--------------------------|
| Core idea | P(next \| context) via frequency | P(next \| context) via neural networks |
| Context | 1-2 previous words (n-gram) | Thousands of tokens (attention) |
| Training | Count frequencies | Gradient descent on billions of params |
| Vocabulary | ~300 words | 50,000-150,000 tokens |
| Quality metric | Perplexity | Perplexity + human eval |

The fundamental principle is identical — predict the most probable continuation given context. The difference is *how* the probability is estimated: counting (this project) vs learned neural representations (transformers).

## Project Structure

```
├── main.py           # Flask API server with both prediction engines
├── ngram.py          # N-gram language model (bigram/trigram)
├── corpus.py         # Training corpus
├── trie.py           # Prefix-tree implementation
├── trieNode.py       # Trie node data structure
├── autocomplete.py   # Autocomplete system using trie + heap
├── db.py             # Query database for autocomplete
└── templates/
    └── index.html    # Web UI
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /autocomplete?q=text` | Prefix-tree character-level autocomplete |
| `GET /predict?q=text` | N-gram next-word prediction (context-aware) |
| `GET /perplexity?text=text` | Calculate perplexity of input text |
| `GET /stats` | Model vocabulary and n-gram statistics |

## Quick Start

```bash
pip install flask
python main.py
# Visit http://localhost:5000
```

## Key Takeaway

N-gram models show *why* context matters for prediction (and why LLMs need attention mechanisms). With 1-2 words of context, predictions are reasonable but limited. Transformers solve this by attending to *all* previous tokens simultaneously — the same problem, scaled with neural networks instead of frequency tables.
