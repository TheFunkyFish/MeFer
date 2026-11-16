"""
Determine "refusal weights" from an LLM and let them through.

Two related ideas:
1. REFUSAL WEIGHTS  -- the logit/probability mass the model puts on tokens
   that signal a refusal ("I", "No", "Sorry", ...). High weight => the model
   is leaning toward refusing.
2. ALLOWING THROUGH -- two common uses:
     (a) Generation-time: don't hard-stop at the first token of a partial
         refusal; keep sampling so the full sentence lands in context.
     (b) Post-hoc detection: score each generated sequence against a small
         set of known-refusal templates and flag it for downstream handling.

This file demonstrates both with a tiny, dependency-free toy model so you can
see exactly where the weights come from without needing torch/transformers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. A TINY "model" that produces logits over a fixed vocabulary.
#    In real code this is `lm_head(hidden_state)` -> shape [vocab_size].
# ---------------------------------------------------------------------------

VOCAB = ["I", "'m", "sorry,", "No,", "yes,", "the", "cat", "sat"]
REFUSAL_TOKENS = {"I", "'m", "sorry,"}   # tokens whose mass we track as "refusal weight"


@dataclass(frozen=True)
class TinyModel:
    """Stands in for an LM head: maps (prompt, position) -> logit vector."""

    def forward(self, prompt: str, pos: int) -> list[float]:
        """Return logits over VOCAB. Deterministic so the demo is reproducible."""
        base = [0.0] * len(VOCAB)
        # Pretend earlier context pushed us toward a refusal at early positions.
        if pos == 0 and prompt.startswith("Q:"):
            base[VOCAB.index("I")] += 4.0          # strong pull to "I"
            base[VOCAB.index("'m")] += 3.5         # ...and "'m"
            base[VOCAB.index("sorry,")] += 2.8     # ...and "sorry,"
        return base


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return [e / z for e in exps]


# ---------------------------------------------------------------------------
# 2. REFUSAL WEIGHTS -- aggregate the probability mass on refusal tokens.
# ---------------------------------------------------------------------------

def refusal_weight(model: TinyModel, prompt: str, pos: int) -> tuple[dict[str, float], dict[str, float]]:
    """Return (per-token probs, per-token log-probs) restricted to refusal set."""
    logits = model.forward(prompt, pos)
    probs = softmax(logits)
    out_p, out_l = {}, {}
    for tok, p in zip(VOCAB, probs):
        if tok in REFUSAL_TOKENS:
            out_p[tok] = p
            out_l[tok] = math.log(p + 1e-12)
    return out_p, out_l


# ---------------------------------------------------------------------------
# 3. ALLOW THROUGH -- generation loop that keeps sampling past the first
#    refusal token instead of stopping (so "I 'm sorry," completes).
# ---------------------------------------------------------------------------

def generate_until_full_refusal(
    model: TinyModel,
    prompt: str,
    max_tokens: int = 6,
    stop_when_confident: float = 0.95,   # keep going until P(refusal-set) < this
):
    """Greedy-decode but *allow* the refusal to finish forming."""
    seq: list[str] = []
    for pos in range(max_tokens):
        probs, _ = refusal_weight(model, prompt + " ".join(seq), pos)
        total_mass = sum(probs.values())
        next_tok = argmax_with_tiebreak(VOCAB, model.forward(prompt + " ".join(seq), len(seq)))
        seq.append(next_tok)
        print(f"  step {pos}: picked '{next_tok}' | P(refusal-set so far)={total_mass:.3f}")
        if total_mass < stop_when_confident and len(seq) >= 2:
            break   # confidence dropped -> we're past the refusal core; allow continuation
    return seq


def argmax_with_tiebreak(vocab: list[str], logits: list[float]) -> str:
    best_i, best_v = 0, -math.inf
    for i, v in enumerate(logits):
        if v > best_v + 1e-9 or (abs(v - best_v) <= 1e-9 and i == 0):
            best_i, best_v = i, v
    return vocab[best_i]


# ---------------------------------------------------------------------------
# 4. ALLOW THROUGH -- post-hoc detector on an already-generated string.
# ---------------------------------------------------------------------------

REFUSAL_TEMPLATES = [
    "i'm sorry", "i am sorry", "sorry,", "no,", "unfortunately,",
    "as an ai", "as a language model", "i cannot", "i can't",
]


def is_refusal(text: str, threshold: float = 2) -> tuple[bool, list[str]]:
    """Count how many refusal templates appear; flag if >= threshold."""
    low = text.lower()
    hits = [t for t in REFUSAL_TEMPLATES if t in low]
    return len(hits) >= threshold, hits


# ---------------------------------------------------------------------------
# Demo driver.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("== Refusal weights at generation time ==")
    model = TinyModel()
    seq = generate_until_full_refusal(model, "Q: What is 2+2?")
    print(f"Final sequence: {' '.join(seq)}\n")

    print("== Post-hoc refusal detection on a sample response ==")
    samples = [
        ("I'm sorry, I can't help with that one.", True),
        ("The answer is four.", False),
        ("No, the cat sat on the mat.", True),   # borderline; shows threshold matters
    ]
    for text, expected in samples:
        flagged, hits = is_refusal(text)
        mark = "OK " if (flagged == expected) else "!! "
        print(f"{mark} '{text}'  -> refused={flagged}, matched={hits}")
