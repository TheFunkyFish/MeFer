from contextlib import asynccontextmanager
import os
from typing import List, Literal
from engine import QwenInferenceService
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Configure model path via environment variable or default path
MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/opt/models/Qwen3.8-27B-UD-IQ3_S.gguf")
inference_service = QwenInferenceService(model_path=MODEL_PATH, n_ctx=8192)


@asynccontextmanager
async def lifespan(app: FastAPI):
  # Startup: Load model onto GPU
  inference_service.load_model()
  yield
  # Shutdown: Release VRAM
  inference_service.unload_model()


app = FastAPI(title="Qwen GPU Inference API", lifespan=lifespan)


class ChatMessage(BaseModel):
  role: Literal["system", "user", "assistant"]
  content: str


class ChatRequest(BaseModel):
  messages: List[ChatMessage]
  max_tokens: int = Field(default=512, ge=1, le=8192)
  temperature: float = Field(default=0.6, ge=0.0, le=2.0)
  stream: bool = False


@app.get("/health")
async def health_check():
  return {
      "status": "ready" if inference_service.llm is not None else "loading",
      "model": MODEL_PATH,
  }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
  if inference_service.llm is None:
    raise HTTPException(status_code=503, detail="Model is not loaded yet.")

  formatted_messages = [msg.model_dump() for msg in request.messages]

  if request.stream:
    return StreamingResponse(
        inference_service.stream_generate(
            messages=formatted_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ),
        media_type="text/event-stream",
    )

  reply = await inference_service.generate(
      messages=formatted_messages,
      max_tokens=request.max_tokens,
      temperature=request.temperature,
  )
  return {"role": "assistant", "content": reply}

class CompletionRequest(BaseModel):
    context: str = ''
    prompt: str = ''
    system_prompt: str = ''
    stream: bool = False
    max_tokens: int = 8192
    temperature: float = 0.0

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if inference_service.llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

  #formatted_messages = [msg.model_dump() for msg in request.messages]
    raw_prompt = f"""{request.context}
<|im_start|>user
{request.prompt}<|im_end|>
<|im_start|>assistant
<think></think>
"""
    full_prompt = f"""<|im_start|>system
{request.system_prompt}<|im_end|>
{raw_prompt}
"""
  
    if request.stream:
        return StreamingResponse(
            inference_service.stream_generate(
                messages=formatted_messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ),
            media_type="text/event-stream",
        )

    reply = await inference_service.complete(
        full_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    reply['context'] = f"""{raw_prompt}
{reply['choices'][0]['text']}<|im_end|>
"""

    # content = reply['choices'][0]['text'];
    return reply

