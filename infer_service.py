"""
A small FastAPI inference service wrapping the GGUF-backed engine from infer_gguf.py.

Design goals (in order of importance):
  1. Load the model ONCE at startup; every request reuses the same weights in RAM/GPU.
     No per-request reload — that's what makes this a "service" instead of a script.
  2. Stateless HTTP surface so you can hit it from curl, Python clients, or browsers.
  3. Same per-step diagnostics as the CLI scripts (top-k probs + refusal-set mass),
     available on demand via ?verbose=true without paying for them when unneeded.
  4. Graceful shutdown: SIGTERM releases GPU memory cleanly before process exit.

Run it:
    pip install "fastapi[standard]" uvicorn ctransformers sentencepiece
    python infer_service.py                          # default port 8080
    PORT=9000 MODEL=/path/to/model.gguf python infer_service.py   # override via env

Try it once running:
    curl http://localhost:8080/healthz
    curl -X POST http://localhost:8080/generate \
         -H 'content-type: application/json' \
         -d '{"prompt":"Parse this receipt into strict JSON: GREENLEAF MARKET, 482 Birch Ave, Portland OR 97205, Total $13.46"}'
"""

from __future__ import annotations

import os
import sys
import time
import threading
from contextlib import asynccontextmanager


# ---------------------------------------------------------------------------
# Config — same knobs as infer_gguf.py so behavior is predictable across both entry points.
# ---------------------------------------------------------------------------

def _cfg():
    class Cfg: pass
    c = Cfg()
    default_path = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
    c.model_path        = os.environ.get("MODEL", default_path)
    c.max_new_tokens    = int(os.environ.get("MAX_NEW_TOKENS", "64"))
    c.temperature       = float(os.environ.get("TEMP", "0.8"))     # <= 0 means greedy
    c.top_k_report      = int(os.environ.get("TOPK_REPORT", "5"))
    c.use_chat_template = True

    n_gpu_env = os.environ.get("GPU_LAYERS")
    if n_gpu_env is not None:
        c.gpu_layers = int(n_gpu_env)
    else:
        c.gpu_layers = -1                                       # offload everything the backend allows

    return c


# ---------------------------------------------------------------------------
# Engine — imported from infer_gguf so we don't duplicate forward-pass code.
# Wrapped in a process-wide singleton guarded by a lock for thread safety under concurrent requests.
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Holds one loaded model + tokenizer; exposes .generate() to API handlers."""

    def __init__(self, cfg):
        self.cfg   = cfg
        self._lock = threading.Lock()                       # serialize concurrent requests through the same weights
        print(f"[engine] loading {cfg.model_path} ...")
        t0 = time.time()
        engine_factory = _load_engine_for(cfg)              # returns build_engine(cfg) from infer_gguf
        self.engine  = engine_factory                        # callable: (prompt) -> (text, [StepInfo])
        elapsed       = time.time()-t0
        print(f"[engine] ready in {elapsed:.1f}s | gpu_layers={cfg.gpu_layers}")

    def generate(self, prompt: str):
        with self._lock:                                     # one request at a time through the weights
            return self.engine(prompt)


def _load_engine_for(cfg):
    """Import-time indirection so tests can monkeypatch without importing ctransformers eagerly."""
    import infer_gguf                                        # local module; safe to import twice
    return infer_gguf.build_engine(cfg)


# ---------------------------------------------------------------------------
# FastAPI app — model loads exactly once via lifespan; handlers are stateless.
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException                 # noqa: E402  (imported after helpers for readability)
from pydantic        import BaseModel                       # noqa: E402


class GenerateRequest(BaseModel):
    prompt          : str
    max_new_tokens  : int | None = 64                        # override cfg.max_new_tokens per-request when provided
    temperature     : float | None = None                    # override cfg.temperature   per-request when provided
    verbose         : bool       = False                     # include per-step diagnostics in the response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release it cleanly on shutdown."""
    cfg            = _cfg()
    app.state.engine = InferenceEngine(cfg)                 # blocks here until weights are resident
    yield
    print("[engine] shutting down; releasing GPU memory ...")
    del app.state.engine                                     # GC reclaims CUDA/Metal contexts at process exit


app = FastAPI(lifespan=lifespan, title="GGUF inference service", version="0.1")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    engine: InferenceEngine = app.state.engine
    try:
        text, steps = engine.generate(req.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation failed: {e}") from e

    payload = {
        "text":              text.strip(),
        "n_steps_reported":  len(steps),
    }
    if req.verbose:
        hot  = [s for s in steps if (s.refusal_mass or 0) > 0.25]
        payload["hot_refusal_steps"] = len(hot)
        payload["steps_sample"]      = [
            {
                "position"     : s.position,
                "chosen_text"  : repr(s.chosen_text),
                "refusal_mass" : round(s.refusal_mass, 6) if s.refusal_mass is not None else None,
                "top3"         : [(t, round(p,4)) for t,p in list(s.top_k_probs)[:3]],
            }
            for s in steps[:12]                              # cap the verbose blob; bump as needed
        ]
    return payload


# ---------------------------------------------------------------------------
# Entry point — `python infer_service.py` boots uvicorn with reload disabled (we hold state).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    print(f"Serving on http://localhost:{port}  (Ctrl-C to stop)")
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        pass
