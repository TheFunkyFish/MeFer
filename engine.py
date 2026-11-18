import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List
from llama_cpp import Llama


class QwenInferenceService:

  def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1):
    self.model_path = model_path
    self.n_ctx = n_ctx
    self.n_gpu_layers = n_gpu_layers
    self.llm: Llama | None = None
    # Prevent concurrent GPU context collisions in llama.cpp
    self._lock = asyncio.Lock()

  def load_model(self) -> None:
    """Loads the GGUF model into GPU VRAM."""
    if self.llm is None:
      print(f"Loading model onto GPU: {self.model_path}")
      self.llm = Llama(
          model_path=self.model_path,
          n_gpu_layers=self.n_gpu_layers,
          n_ctx=self.n_ctx,
          verbose=True,
          #split_mode=0,
          #main_gpu=0,
          #tensor_split= [0.8, 0.2],
          #type_k= 0,
          #type_v= 1,
          flash_attn=True
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
      print(f"{usage}")
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
      print(f"Elapsed time: {elapsed_seconds} seconds @ 1 token per {token_generated} second for {1/token_generated} TPS ")
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

