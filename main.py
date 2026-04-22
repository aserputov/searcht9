"""
Statistical Next-Token Prediction Engine

Combines two approaches:
1. Prefix-tree (Trie) autocomplete — fast character-level prefix matching
2. N-gram language model — statistical next-word prediction with context

Demonstrates core concepts underlying modern autoregressive LLMs:
- Frequency-based token prediction (n-gram statistics)
- Context-dependent prediction (bigram/trigram backoff)
- Vocabulary search optimization (trie + heap for top-k)
- Perplexity evaluation (standard LM metric)
"""

from flask import Flask, request, jsonify, render_template
from autocomplete import AutocompleteSystem
from ngram import NGramModel
from corpus import CORPUS
from db import db

app = Flask(__name__)

# Prefix-tree autocomplete (character-level)
autocomplete_system = AutocompleteSystem(db[0], db[1])

# N-gram language model (word-level, context-aware)
ngram_model = NGramModel(n=3)
ngram_model.train(CORPUS)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    """Prefix-tree based autocomplete (character-level matching)."""
    query = request.args.get("q", "")
    suggestions = autocomplete_system.input(query)
    autocomplete_system.input("#")
    return jsonify(suggestions)


@app.route("/predict", methods=["GET"])
def predict():
    """N-gram next-token prediction (word-level, context-aware)."""
    context = request.args.get("q", "")
    top_k = int(request.args.get("k", 5))
    predictions = ngram_model.predict_next(context, top_k=top_k)
    return jsonify({
        "context": context,
        "predictions": [
            {"token": token, "probability": round(prob, 4)}
            for token, prob in predictions
        ],
    })


@app.route("/stats", methods=["GET"])
def stats():
    """Return model statistics."""
    return jsonify(ngram_model.stats())


@app.route("/perplexity", methods=["GET"])
def perplexity():
    """Calculate perplexity of input text."""
    text = request.args.get("text", "")
    if not text:
        return jsonify({"error": "provide ?text= parameter"}), 400
    ppl = ngram_model.perplexity(text)
    return jsonify({"text": text, "perplexity": round(ppl, 2)})


if __name__ == "__main__":
    stats_data = ngram_model.stats()
    print(f"N-gram model trained: {stats_data['vocab_size']} vocab, "
          f"{stats_data['unique_bigrams']} bigrams, {stats_data['unique_trigrams']} trigrams")
    print("Endpoints:")
    print("  /autocomplete?q=the    (prefix-tree character matching)")
    print("  /predict?q=the cat     (n-gram next-word prediction)")
    print("  /perplexity?text=...   (evaluate text probability)")
    app.run(debug=True)
