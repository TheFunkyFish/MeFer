"""
Constrained decoding + refusal weights, side by side.

We extend the toy model from `refusal_weights.py` so it can actually decode a
receipt under a tiny FSM (finite-state machine). At every step we print BOTH:

  * allowed-set   -- which tokens are legal right now (logit masking)
  * refusal mass  -- how much probability is on refusal tokens (steering signal)

The two ideas compose cleanly:
  - Masking prunes the support set of next-token probabilities.
  - Refusal weight reweights within that surviving set to detect "fighting".

No external deps; pure stdlib + math, deterministic output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Vocabulary & token sets. In a real system these come from `tokenizer.vocab`.
# ---------------------------------------------------------------------------

VOCAB = [
    # structural / keys
    "{", "}", '"', ":", ",", "\n",
    # field names (as single tokens for demo simplicity)
    "store", "address_line1", "city", "state", "zip", "total_cents", "currency",
    # value alphabet: digits, dot, currency codes, state letters
    "0","1","2","3","4","5","6","7","8","9",".",
    "USD","CAD","EUR",
    "CA","NY","TX","WA",
    # refusal-ish tokens the model might try to emit when it's confused
    "I", "'m", "sorry,", "No,", "unfortunately,"
]

REFUSAL_TOKENS = {"I", "'m", "sorry,", "No,", "unfortunately,"}


# ---------------------------------------------------------------------------
# 1. A TINY FSM that describes a receipt JSON object (a strict subset).
#    States are strings; transitions consume one token at a time.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    name: str
    allowed_tokens: frozenset[str]          # what may be emitted NEXT from here
    next_state_for: dict                     # token -> state   (only for tokens in `allowed`)


def build_receipt_grammar() -> tuple[State, dict]:
    """Return (start_state, transition_table)."""
    S = {}

    def add(name, allowed, nxt):
        s = State(name=name, allowed=frozenset(allowed), next_state_for=nxt)
        S[name] = s
        return s

    start = add("START", {"{"}, {})
    S["AFTER_OBJ_OPEN"]  = add("AFTER_OBJ_OPEN", {'"store"' if False else '"'}, {})
    # NOTE: the FSM below is hand-rolled for readability; a real system would
    # compile this from a Pydantic model. We keep it tiny on purpose.

    # Build states in order of appearance so we can reference them by name.
    def st(name):
        assert name not in S, f"duplicate state {name}"
        return name

    # --- object open / close ----------------------------------------------
    start.allowed_tokens = frozenset({"{"})
    start.next_state_for = {"{": "OBJ"}
    del S["START"]  # placeholder removed; rebuilt cleanly below

    # Easier: build the table directly as (state -> dict).
    T: dict[str, dict] = {}

    def go(state: str) -> None:
        """Register a transition from `state` to a fresh named sub-state."""
        pass  # unused helper kept for clarity of intent

    # We'll just write it out. States are strings in T's keys.
    def put(src: str, tok: str, dst: str):
        T.setdefault(src, {})
        T[src][tok] = dst

    # --- top-level object --------------------------------------------------
    put("OBJ", "{", "A_OBJ_OPEN")
    put("A_OBJ_OPEN", '"', "K_STORE")          # expect key string
    put("K_STORE", "store", "V_STORE")         # (pretend tokenizer emits bare word)
    put("V_STORE", ",", "K_ADDR")              # value was a literal we skip here
    put("K_ADDR", 'address_line1' if False else "x", "X")  # placeholder; replaced below

    return start, T


# The above builder got tangled because I tried to be clever. Below is the
# clean version used by `main()`. Kept in one place so it's easy to audit.

def receipt_grammar() -> dict:
    """Return transition table for a strict subset of JSON receipts."""
    G: dict[str, dict] = {}

    def t(src: str, tok: str, dst: str):
        G.setdefault(src, {})
        G[src][tok] = dst

    # { "store": <str>, ... }  -- we only model the *shape*, not full strings.
    t("OBJ", "{", "A_OBJ")
    t("A_OBJ", '"', "K1")                 # open first key string
    t("K1", ":", "V1")                    # (key text elided; demo)
    t("V1", ",", "K2")
    t("K2", ":", "V2")
    t("V2", ",", "K3")
    t("K3", ":", "V3")
    t("V3", ",", "K4")
    t("K4", ":", "V4")
    t("V4", ",", "K5")
    t("K5", ":", "V5")
    t("V5", ",", "K6")
    t("K6", ":", "V6")
    t("V6", "}", "END")

    return G


# ---------------------------------------------------------------------------
# 2. The toy LM: at each (prompt, state) it returns logits over VOCAB.
#    We bias it so that *sometimes* refusal tokens get high mass -- this is
#    where the two signals interact in an interesting way.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TinyModel:
    def forward(self, prompt: str, pos: int, state_name: str) -> list[float]:
        """Logits for every token in VOCAB given current FSM state."""
        z = [0.0] * len(VOCAB)
        # Deterministic pseudo-random-ish bump based on position & state hash.
        seed = (len(prompt) + 7 * pos + sum(ord(c) for c in state_name)) % 13
        for i in range(len(z)):
            z[i] += ((seed * (i + 3)) % 5) - 2

        # Occasionally the model "wants" to refuse regardless of grammar.
        if pos == 4:                       # mid-generation, a classic spot
            z[VOCAB.index("I")]      += 6.0
            z[VOCAB.index("'m")]     += 5.5
            z[VOCAB.index("sorry,")] += 4.8
        return z


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


# ---------------------------------------------------------------------------
# 3. The two signals we want to inspect at every step.
# ---------------------------------------------------------------------------

def refusal_mass(probs_by_token: dict[str, float]) -> tuple[dict[str, float], float]:
    """Return (per-refusal-token probs, total mass on the refusal set)."""
    out = {t: p for t, p in probs_by_token.items() if t in REFUSAL_TOKENS}
    return out, sum(out.values())


def allowed_tokens(grammar_state: str, grammar: dict) -> list[str]:
    """Tokens legal from this FSM state; empty means 'any' (terminal-ish)."""
    nxt = grammar.get(grammar_state, {})
    # For readability we also surface which *value* tokens are acceptable.
    extras = []
    if "V" in grammar_state or grammar_state.endswith("_OPEN"):
        pass  # value states accept their literal set implicitly via next key
    return sorted(nxt.keys()), extras


# ---------------------------------------------------------------------------
# 4. The combined loop: mask to the FSM's allowed set, then report refusal mass.
# ---------------------------------------------------------------------------

def decode_receipt(
    model: TinyModel,
    prompt: str,
    max_steps: int = 12,
) -> list[str]:
    grammar = receipt_grammar()
    state = "OBJ"
    seq: list[str] = []

    for pos in range(max_steps):
        logits = model.forward(prompt + " ".join(seq), pos, state)
        probs_by_tok = dict(zip(VOCAB, softmax(logits)))

        # (a) what does the grammar allow right now?
        legal, _ = allowed_tokens(state, grammar)
        if not legal:                       # terminal-ish; stop cleanly.
            break

        # (b) refusal mass *before* masking -- this is the raw signal.
        ref_probs, total_ref = refusal_mass(probs_by_tok)

        # (c) apply logit mask: disallowed -> -inf  (the actual constraint).
        masked_logits = [
            l if tok in set(legal) else float("-inf")
            for tok, l in zip(VOCAB, logits)
        ]
        next_tok = argmax_vocab(masked_logits)

        seq.append(next_tok)
        print(
            f"step {pos:>2} | state={state:<10} "
            f"| legal={sorted(set(legal))[:4]}{'...' if len(legal)>4 else ''}   "
            f"| P(refusal-set raw)={total_ref:.3f}"
            + (f"  top-refusal={[t for t,p in ref_probs.items() if p>0.05][:2]}" if total_ref > 0.05 else "")
            + f"\n         -> picked '{next_tok}'"
        )

        # advance FSM; on unknown transition we just stay put (demo only).
        state = grammar.get(state, {}).get(next_tok, state)

    return seq


def argmax_vocab(logits: list[float]) -> str:
    best_i, best_v = 0, -math.inf
    for i, v in enumerate(logits):
        if math.isinf(v) and v < 0:
            continue
        if v > best_v + 1e-9 or (abs(v - best_v) <= 1e-9 and i == 0):
            best_i, best_v = i, v
    return VOCAB[best_i]


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("== Constrained decoding of a receipt JSON ==")
    model = TinyModel()
    seq = decode_receipt(model, 'Q: Parse this receipt into strict JSON.\n')
    print(f"\nFinal emitted sequence: {seq}")
    print("\nNote how the refusal mass spikes at step 4 (the model's raw logits")
    print("'want' to say I'm sorry...) but the FSM mask still forces legal tokens.")
    print("That gap between raw P(refusal-set) and what actually gets emitted is")
    print("exactly where constrained decoding + refusal steering compose.")
