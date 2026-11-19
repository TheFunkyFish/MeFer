import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List
from llama_cpp import Llama

class QwenInferenceService:

  def __init__(self, cfg):
    self.llm: Llama | None = None
    self.cfg = cfg
    # Prevent concurrent GPU context collisions in llama.cpp
    self._lock = asyncio.Lock()

  def load_model(self) -> None:
    """Loads the GGUF model into GPU VRAM."""
    if self.llm is None:
      print(f"Loading model onto GPU: {self.cfg['model_path']}")
      self.llm = Llama(
        model_path=self.cfg.get("model_path", "/opt/models/Qwen3.8-27B-UD-IQ3_S.gguf"),
        n_gpu_layers=self.cfg.get("n_gpu_layers", -1),  # -1 offloads all layers to GPU
        split_mode=self.cfg.get("split_mode", 1),  # 1 = LLAMA_SPLIT_MODE_LAYER
        main_gpu=self.cfg.get("main_gpu", 0),
        tensor_split=self.cfg.get("tensor_split", None),
        rpc_servers=self.cfg.get("rpc_servers", None),
        vocab_only=self.cfg.get("vocab_only", False),
        use_mmap=self.cfg.get("use_mmap", True),
        use_mlock=self.cfg.get("use_mlock", False),
        check_tensors=self.cfg.get("check_tensors", False),
        kv_overrides=self.cfg.get("kv_overrides", None),

        # --- Context & Batching (Tuned for Qwen 3.8) ---
        # Native context is 262,144; 32,768–65,536 balances VRAM and reasoning depth
        n_ctx=self.cfg.get("n_ctx", 32768),
        n_batch=self.cfg.get("n_batch", 2048),
        n_ubatch=self.cfg.get("n_ubatch", 512),
        n_threads=self.cfg.get("n_threads", 4),
        n_threads_batch=self.cfg.get("n_threads_batch", 4),

        # --- RoPE Scaling (YaRN for >262k Context) ---
        # Set rope_scaling_type=2 (YaRN) and yarn_orig_ctx=262144 if extending past 262k
        rope_scaling_type=self.cfg.get("rope_scaling_type", -1),
        pooling_type=self.cfg.get("pooling_type", -1),
        attention_type=self.cfg.get("attention_type", -1),
        flash_attn=self.cfg.get("flash_attn", True),  # Essential for Qwen 3.8 speed/VRAM
        rope_freq_base=self.cfg.get("rope_freq_base", 0.0),
        rope_freq_scale=self.cfg.get("rope_freq_scale", 0.0),
        yarn_ext_factor=self.cfg.get("yarn_ext_factor", -1.0),
        yarn_attn_factor=self.cfg.get("yarn_attn_factor", 1.0),
        yarn_beta_fast=self.cfg.get("yarn_beta_fast", 32.0),
        yarn_beta_slow=self.cfg.get("yarn_beta_slow", 1.0),
        yarn_orig_ctx=self.cfg.get("yarn_orig_ctx", 262144),

        # --- KV Cache & Memory Precision ---
        # Keep KV cache at f16 (type=1) to preserve Qwen 3.8 reasoning quality
        type_k=self.cfg.get("type_k", 1),
        type_v=self.cfg.get("type_v", 1),
        offload_kqv=self.cfg.get("offload_kqv", True),
        no_perf=self.cfg.get("no_perf", False),

        # --- Generation, Chat & Adapters ---
        logits_all=self.cfg.get("logits_all", False),
        embedding=self.cfg.get("embedding", False),
        seed=self.cfg.get("seed", -1),  # -1 uses a random seed per run
        max_tokens=self.cfg.get("max_tokens", 8192),
        last_n_tokens_size=self.cfg.get("last_n_tokens_size", 64),
        lora_base=self.cfg.get("lora_base", None),
        lora_scale=self.cfg.get("lora_scale", 1.0),
        lora_path=self.cfg.get("lora_path", None),
        numa=self.cfg.get("numa", False),
        chat_format=self.cfg.get("chat_format", None),  # None auto-detects ChatML from GGUF
        chat_handler=self.cfg.get("chat_handler", None),
        draft_model=self.cfg.get("draft_model", None),
        tokenizer=self.cfg.get("tokenizer", None),
        verbose=self.cfg.get("verbose", False),
      
      )
      print("Model successfully loaded into VRAM.")

  def unload_model(self) -> None:
    """Frees GPU VRAM on server shutdown."""
    if self.llm is not None:
      del self.llm
      self.llm = None
      print("Model unloaded from VRAM.")

  async def generate(
      self,
      messages: List[Dict[str, str]],
      max_tokens: int = 512,
      temperature: float = 0.6,
  ) -> str:
    """Runs synchronous GPU inference in a worker thread without blocking FastAPI."""
    if self.llm is None:
      raise RuntimeError("Model is not loaded.")

    async with self._lock:
      response = await asyncio.to_thread(
          self.llm.create_chat_completion,
          messages=messages,
          max_tokens=max_tokens,
          temperature=temperature,
          stream=False,
      )
      usage = response.get("usage", {})
      #print(f"{usage}")
      return response["choices"][0]["message"]["content"]

  async def complete(
      self,
      prompt,
      max_tokens: int = 8192 * 16,
      temperature: float = 0.6,

  ) -> str:
    """Runs synchronous GPU inference in a worker thread without blocking FastAPI."""
    if self.llm is None:
      raise RuntimeError("Model is not loaded.")
    start_time = time.perf_counter()
    async with self._lock:
      response = await asyncio.to_thread(
          self.llm.create_completion,
          prompt=prompt,
          max_tokens=max_tokens,
          temperature=temperature,
          stream=False
      )
      usage = response.get("usage", {})
      end_time = time.perf_counter()
      elapsed_seconds = end_time - start_time
      token_generated = elapsed_seconds/usage.get('total_tokens', 1)
      #print(f"Elapsed time: {elapsed_seconds} seconds @ 1 token per {token_generated} second for {1/token_generated} TPS ")
      print(f"{response}")
      return response

  async def stream_generate(
      self,
      messages: List[Dict[str, str]],
      max_tokens: int = 512,
      temperature: float = 0.6,
  ) -> AsyncGenerator[str, None]:
    """Yields Server-Sent Events (SSE) chunks as tokens are generated on the GPU."""
    if self.llm is None:
      raise RuntimeError("Model is not loaded.")

    async with self._lock:
      stream = self.llm.create_chat_completion(
          messages=messages,
          max_tokens=max_tokens,
          temperature=temperature,
          stream=True,
      )

      def _next_chunk():
        return next(stream, None)

      while True:
        chunk = await asyncio.to_thread(_next_chunk)
        if chunk is None:
          break
        delta = chunk["choices"][0]["delta"]
        if "content" in delta and delta["content"]:
          payload = json.dumps({"content": delta["content"]})
          yield f"data: {payload}\n\n"

      yield "data: [DONE]\n\n"

