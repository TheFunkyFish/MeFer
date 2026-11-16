"""
A minimal inference engine for small LLaMA-style Instruct models backed by a
GGUF file (llama.cpp / ctransformers runtime).

Same shape as `infer.py` so you can diff them side-by-side:
  - per-step diagnostics (top-k probs, refusal-set mass) at every token
  - optional logit masking hook for constrained decoding
  - env-var config so you don't have to edit the file between runs

The only real difference from infer.py is where logits come from: instead of an
HF model object we drive a GGUF-backed forward pass through `ctransformers`.
Everything downstream — softmax, top-k reporting, refusal-mass tracking, chat-
template rendering — stays in Python so it's easy to audit.

Run it:
    python infer_gguf.py                          # uses default path below
    MODEL=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf \
        MAX_NEW_TOKENS=48 TEMP=0.7 python infer_gguf.py

Requirements (one-time):
    pip install ctransformers sentencepiece
    # or, if you prefer the llama-c-python binding instead:
    #   pip install llama-c-python
"""

from __future__ import annotations

import os
import sys
import time


# ---------------------------------------------------------------------------
# Config — knobs in one place; overridable from env for quick iteration.
# ---------------------------------------------------------------------------

def _detect_gpu():
    """Return (backend_name, device_count). backend is one of: 'metal', 'cuda', 'cpu'."""
    import platform
    sysname = platform.system().lower()          # 'darwin' | 'linux' | 'windows'
    if sysname == "darwin":
        # Apple Silicon unified memory shows up as a Metal GPU to llama.cpp.
        try:
            import ctypes
            # Cheap probe: importing the framework succeeds only when present.
            import metal_3  # noqa: F401 — optional; not required for detection below
        except ImportError:
            pass
        return ("metal", 1)                     # single unified pool on M-series / Intel Macs w/ discrete dGPU
    # NVIDIA CUDA path (Linux + Windows). We don't hard-import torch here so we can
    # still run this file in environments where it isn't installed yet.
    try:
        import subprocess, shutil
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            return ("cpu", 0)
        out = subprocess.check_output([nvidia_smi, "--query-gpu=name,memory.total", "-N=1"], text=True)
        first_line = out.strip().splitlines()[0]
        name = first_line.split("|")[0].strip() or "cuda"
        mem_mb   = int(first_line.split("|")[1].strip().replace(" MiB","").strip())
        return (f"cuda:{name}", max(1, min(mem_mb // (2*1024), 8)))    # rough layer budget hint
    except Exception:
        pass
    return ("cpu", 0)


def _cfg():
    class Cfg: pass
    c = Cfg()
    default_path = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
    c.model_path        = os.environ.get("MODEL", default_path)
    c.max_new_tokens    = int(os.environ.get("MAX_NEW_TOKENS", "64"))
    c.temperature       = float(os.environ.get("TEMP", "0.8"))     # <= 0 means greedy
    c.top_k_report      = int(os.environ.get("TOPK_REPORT", "5"))
    c.use_chat_template = True

    # --- GPU selection ------------------------------------------------------
    backend, hint_count = _detect_gpu()
    if os.environ.get("GPU_LAYERS") is not None:
        c.gpu_layers = int(os.environ["GPU_LAYERS"])             # explicit override wins
    else:
        # Default: put as many layers on the accelerator as we sensibly can.
        # For a ~135M-param model everything fits comfortably; for larger models
        # you'd want to leave room for KV cache + activations in system RAM.
        c.gpu_layers = 999                                       # "all of them" sentinel

    c.backend_hint      = backend                                 # informational only
    c.suggested_gpu_layers = hint_count                            # what _detect_gpu() guessed
    return c


# ---------------------------------------------------------------------------
# Engine — a thin wrapper over the GGUF runtime that exposes per-step diagnostics.
# ---------------------------------------------------------------------------

def build_engine(cfg):
    """Returns `engine(prompt) -> (full_text, [StepInfo])`.

    StepInfo carries: position, chosen_token_id/text, top-k probs with decoded
    strings, and refusal-set mass BEFORE any temperature or masking is applied.
    That ordering matters: measure first, then constrain/sample.
    """
    from ctransformers import AutoModel as CTRAuto   # deferred so --help works w/o deps

    n_gpu = cfg.gpu_layers if cfg.gpu_layers < 999 else -1     # -1 == "all layers" to most backends
    print(f"Loading {cfg.model_path} ...")
    t0 = time.time()
    model = CTRAuto.from_pretrained(
        cfg.model_path,
        local_files_only=True,
        gpu_layers=n_gpu,                    # -1 => offload everything the backend allows
        dry_run=False,
    )
    tok  = _make_tokenizer(model)             # small shim around the runtime's tokenizer
    refusal_ids: set[int] = _resolve_refusal_token_ids(tok)

    elapsed = time.time()-t0
    where   = getattr(model, "device", None) or getattr(getattr(model, "llama_model", None), "devices", "?")
    print(f"Loaded in {elapsed:.1f}s | device={where} | requested gpu_layers={cfg.gpu_layers}")
    if cfg.suggested_gpu_layers and n_gpu <= 0:
        pass                                  # CPU-only path; nothing to warn about.

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

        input_ids  = list(tok.encode(rendered))
        past_len   = len(input_ids)

        for step in range(cfg.max_new_tokens):
            # --- forward pass over the current window ----------------------
            logits = _forward_window(model, input_ids, max_ctx=256)     # [vocab_size], pre-softmax

            probs_all    = _softmax(logits)
            refusal_mass = (
                float(sum(p.item() for i, p in enumerate(probs_all) if i in refusal_ids))
                if refusal_ids else None
            )

            chosen_id, topk_pairs = _sample(
                logits=logits, temperature=cfg.temperature, report_k=cfg.top_k_report, tok=tok,
            )
            input_ids.append(int(chosen_id))
            steps.append(_StepInfo(
                position=len(input_ids) - 1,
                chosen_token_id=int(chosen_id),
                chosen_text=tok.decode([int(chosen_id)]),
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


def _make_tokenizer(model):
    """Wrap the runtime's tokenizer in a tiny object with .encode/.decode/.chat_template."""
    class TokShim:
        chat_template = getattr(getattr(model, "tokenizer", None), "chat_template", None) or \
                       getattr(model, "chat_template", None)
        def encode(self, text, *a, **k):  return list(model.tokenize(text, as_tensor=False))
        def decode(self, ids, *a, **k):   return model.detokenize(list(ids), skip_special_tokens=True)
    return TokShim()


def _forward_window(model, input_ids: list[int], max_ctx=256):
    """Run one forward pass over the last `max_ctx` tokens and return final-layer logits.

    ctransformers exposes this via `.get_logits(...)` on most builds; if your build
    only offers a generate-style API you can fall back to sampling one token at a
    time with `return_type="logits"` — same numbers, slightly less control.
    """
    window = input_ids[-min(max_ctx, len(input_ids)):]
    try:
        # Preferred path: direct logit access for full per-step visibility.
        out = model.get_logits(window)          # shape [vocab_size] or [1, vocab_size]
        import torch
        return out.reshape(-1).float() if hasattr(out, "reshape") else _as_tensor_1d(out)
    except AttributeError:
        raise RuntimeError(
            "This ctransformers build doesn't expose get_logits(). Either upgrade the "
            "runtime or switch to infer.py (HF-backed) which has the same diagnostics."
        )


def _sample(logits, temperature: float, report_k: int, tok):
    """Greedy pick + top-k audit trail. Swap in multinomial sampling when you want stochastic decoding."""
    scaled = logits / max(temperature, 1e-9) if temperature > 0 else logits
    probs   = _softmax(scaled)

    k      = min(report_k, probs.numel())
    vals, idxs = torch_topk(probs, k=k)
    topk_pairs = []
    for i, p in zip(idxs.tolist(), vals.tolist()):
        try:
            txt = " ".join(tok.decode([i]).split()) or f"<id={i}>"
        except Exception:
            txt = f"<id={i}>"
        topk_pairs.append((txt, float(p)))

    chosen = int(torch_argmax(logits))          # greedy; swap with multinomial for sampling
    return chosen, topk_pairs


def _softmax(x):
    import torch
    m  = x.max(dim=-1, keepdim=True).values
    e  = (x - m).exp()
    return e / e.sum(-1, keepdim=True)


# Tiny shims so the file runs even if `torch` isn't imported at module scope.
try:
    import torch as _t
except ImportError:
    _t = None

def torch_topk(t, k=5):
    if _t is not None and isinstance(t, _t.Tensor):
        v, i = t.topk(k=k);  return v, i
    # Pure-Python fallback for numpy-ish arrays returned by some ctransformers builds.
    import operator
    flat = list(enumerate(t.tolist() if hasattr(t, "tolist") else t))
    top  = sorted(flat, key=lambda kv: -kv[1])[:k]
    class V: pass
    v = V(); v.tolist = lambda: [p for _, p in top]
    i = V(); i.tolist = lambda: [idx for idx, _ in top]
    return v, i

def torch_argmax(t):
    if _t is not None and isinstance(t, _t.Tensor):
        return int(_t.argmax(t).item())
    seq = t.tolist() if hasattr(t, "tolist") else list(t)
    best_i, best_v = 0, -float("inf")
    for i, v in enumerate(seq):
        if v > best_v + 1e-9 or (abs(v - best_v) <= 1e-9 and i == 0):
            best_i, best_v = i, v
    return best_i


def _as_tensor_1d(x):
    import torch
    return torch.as_tensor(x, dtype=torch.float32).reshape(-1)


def _resolve_refusal_token_ids(tok) -> set[int]:
    """Map common refusal phrases to their token IDs so we can track mass live."""
    candidates = ["I'm sorry", "i am sorry", "Sorry,", "No,", "unfortunately"]
    ids: set[int] = set()
    for c in candidates:
        try:
            enc = tok.encode(c)
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
    print(f"Model file : {cfg.model_path}")
    if cfg.suggested_gpu_layers > 0 or "cuda" in (cfg.backend_hint or ""):
        print(f"GPU hint   : backend={cfg.backend_hint} suggested_layers~{cfg.suggested_gpu_layers}"
              f"  (override with GPU_LAYERS=<n>)")
    elif cfg.backend_hint == "metal":
        print("GPU hint   : Metal detected — all layers will be offloaded by default.")
    else:
        print("GPU hint   : no accelerator detected; running CPU-only. Install a CUDA/Metal build of ctransformers for speedups.")
    engine = build_engine(cfg)

    try:
        text, steps = engine(RECEIPT_PROMPT)
    except Exception as e:
        print(f"[error] generation failed: {e}", file=sys.stderr)
        raise

    print("\n=== Generated response ===")
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
