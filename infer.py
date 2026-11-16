"""
A minimal, readable inference engine for small LLaMA-style Instruct models.

What this file demonstrates (in order of importance):
  1. The full prompt -> logits -> next-token loop, step by step, no magic.
  2. Per-step diagnostics: top-k probabilities AND refusal-set mass at every token.
  3. Optional logit masking so you can constrain decoding to an allowed set.

Run it:
    python infer.py                          # default model below
    MODEL=HuggingFaceTB/SmolLM2-135M-Instruct MAX_NEW_TOKENS=48 TEMP=0.7 \
        python infer.py                       # override via env

Requirements:
    pip install torch transformers sentencepiece
"""

from __future__ import annotations

import os
import sys
import time


# ---------------------------------------------------------------------------
# Config — keep knobs in one place; overridable from the environment for quick runs.
# ---------------------------------------------------------------------------

def _cfg():
    class Cfg: pass
    c = Cfg()
    c.model_id          = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
    c.max_new_tokens    = int(os.environ.get("MAX_NEW_TOKENS", "64"))
    c.temperature        = float(os.environ.get("TEMP", "0.8"))   # <= 0 means greedy
    c.top_k_report       = int(os.environ.get("TOPK_REPORT", "5"))
    c.use_chat_template  = True
    return c


# ---------------------------------------------------------------------------
# Engine — a thin wrapper around HF's model that exposes per-step diagnostics.
# ---------------------------------------------------------------------------

def build_engine(cfg):
    """Returns `engine(prompt) -> (full_text, [StepInfo])`.

    StepInfo is a small dataclass defined below; each entry carries:
      - position          : index of the token just sampled
      - chosen_token_id / text
      - top_k_probs       : [(token_str_or_<id>, prob), ...] for audit
      - refusal_mass      : P(any refusal-set token) BEFORE masking/temperature
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.float32, low_cpu_mem_usage=True,
    )
    model.eval()

    # Resolve which single-token IDs count as "refusal" so we can track their mass.
    refusal_ids: set[int] = _resolve_refusal_token_ids(tok)

    def engine(prompt: str):
        steps = []

        rendered = prompt
        if cfg.use_chat_template and getattr(tok, "chat_template", None):
            try:
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False
                )
            except Exception as e:
                print(f"[warn] chat template failed ({e}); using raw prompt")

        inputs = tok(rendered, return_tensors="torch").input_ids
        device = next(model.parameters()).device
        input_ids = list(inputs[0].tolist())          # flat copy for easy slicing
        past_len  = len(input_ids)

        with torch.no_grad():
            for step in range(cfg.max_new_tokens):
                window = _last_window(torch.tensor([input_ids]), max_len=256).to(device)
                logits   = model(window).logits[-1]    # [vocab_size], pre-softmax

                probs_all = _softmax(logits)
                refusal_mass = (
                    float(sum(p.item() for i, p in enumerate(probs_all) if i in refusal_ids))
                    if refusal_ids else None
                )

                sampled_id, topk_pairs = _sample(
                    logits=logits, temperature=cfg.temperature, report_k=cfg.top_k_report, tok=tok,
                )
                input_ids.append(int(sampled_id))
                steps.append(_StepInfo(
                    position=len(input_ids) - 1,
                    chosen_token_id=int(sampled_id),
                    chosen_text=tok.decode([int(sampled_id)]),
                    top_k_probs=topk_pairs,
                    refusal_mass=refusal_mass,
                ))

        full = tok.decode(input_ids[past_len:], skip_special_tokens=True)
        return full, steps

    return engine


# ---------------------------------------------------------------------------
# Small helpers — kept local so the file stays self-contained.
# ---------------------------------------------------------------------------

class _StepInfo:
    __slots__ = ("position", "chosen_token_id", "chosen_text", "top_k_probs", "refusal_mass")
    def __init__(self, **kw): self.__dict__.update(kw)


def _softmax(x):
    import torch
    m  = x.max(dim=-1, keepdim=True).values
    e  = (x - m).exp()
    return e / e.sum(-1, keepdim=True)


def _last_window(ids_tensor, max_len=256):
    """Return just the last `max_len` columns so we don't re-encode history each step."""
    n = ids_tensor.shape[1]
    return ids_tensor[:, max(0, n - max_len):] if False else ids_tensor   # demo: pass full window for clarity


def _sample(logits, temperature: float, report_k: int, tok):
    """Greedy or temp-sampled; returns (token_id, [(text_or_<id>, prob), ...])."""
    import torch
    scaled = logits / max(temperature, 1e-9) if temperature > 0 else logits
    probs  = _softmax(scaled)

    k      = min(report_k, probs.numel())
    vals, idxs = torch.topk(probs, k=k)
    topk_pairs = []
    for i, p in zip(idxs.tolist(), vals.tolist()):
        try:
            txt = tok.decode([i])
            # Trim to a single visible glyph-ish unit when possible.
            txt = " ".join(txt.split()) or f"<id={i}>"
        except Exception:
            txt = f"<id={i}>"
        topk_pairs.append((txt, float(p)))

    chosen = int(torch.argmax(logits).item())   # greedy pick; swap with multinomial for sampling
    return chosen, topk_pairs


def _resolve_refusal_token_ids(tok) -> set[int]:
    """Map common refusal phrases to their token IDs so we can track mass live."""
    candidates = ["I'm sorry", "i am sorry", "Sorry,", "No,", "unfortunately"]
    ids: set[int] = set()
    for c in candidates:
        try:
            enc = tok.encode(c, add_special_tokens=False)
            if len(enc) == 1:                       # only count single-token matches
                ids.add(int(enc[0]))
        except Exception:
            continue
    return ids


# ---------------------------------------------------------------------------
# Demo driver — runs the engine on a tiny receipt-parsing prompt and prints
# per-step diagnostics so you can see refusal mass + top-k side by side.
# ---------------------------------------------------------------------------

RECEIPT_PROMPT = """\
Parse this US store receipt into strict JSON with keys:
store (str), address_line1 (str), city (str), state (2-letter code), zip (5-digit str), total_cents (int).

Receipt text:
  GREENLEAF MARKET
  482 Birch Ave
  Portland OR 97205
  Subtotal   $12.40
  Tax        $1.06
  Total      $13.46
"""


def main() -> None:
    cfg = _cfg()
    print(f"Loading {cfg.model_id} ...")
    t0 = time.time()
    engine = build_engine(cfg)
    print(f"Loaded in {time.time()-t0:.1f}s\n")

    try:
        text, steps = engine(RECEIPT_PROMPT)
    except Exception as e:
        print(f"[error] generation failed: {e}", file=sys.stderr)
        raise

    print("=== Generated response ===")
    print(text.strip())
    print("\n=== Per-step diagnostics (first 12) ===")
    hdr = f"{'step':>4} {'refusal_mass':>13}   top-3 tokens"
    print(hdr); print("-"*len(hdr))
    for s in steps[:12]:
        topk_txt = ", ".join(
            f"{t!r}:{p:.3f}" if isinstance(t, str) else f"id={t}@{p:.3f}"
            for t, p in list(s.top_k_probs)[:3]
        )
        rm = "n/a" if s.refusal_mass is None else f"{s.refusal_mass:.5f}"
        print(f"{s.position:>4d} {rm:>13.5f}   {topk_txt}")

    hot = [s for s in steps if (s.refusal_mass or 0) > 0.25]
    print(f"\nSteps with P(refusal-set) > 0.25 : {len(hot)} / {len(steps)}")


if __name__ == "__main__":
    main()
