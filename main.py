from contextlib import asynccontextmanager
import os
from typing import List, Literal
from engine import QwenInferenceService
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from config import ConfigManager
import re
# Configure model path via environment variable or default path
cfg = ConfigManager('config.yaml')
inference_service = QwenInferenceService(cfg)


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

def extract_tool_calls(raw_template: str) -> list[dict]:
    """
    Extract tool calls from a raw Qwen 3.8 template string.
    
    Qwen 3.8 uses a specific format for tool calls within the template.
    Tool calls are typically embedded in the assistant's response using a
    structured format that includes the tool name and arguments.
    
    The expected format in the raw template string looks like:
    <tool_call>
    {"name": "tool_name", "arguments": {"arg1": "value1", ...}}
    </tool_call>
    
    Or potentially in a JSON-like block within the assistant message.
    
    Args:
        raw_template: The raw Qwen 3.8 template string containing potential tool calls.
        
    Returns:
        A list of dictionaries, each representing a tool call with 'name' and 'arguments' keys.
    """
    tool_calls = []
    
    # Pattern to match tool call blocks
    # Qwen 3.8 may use various formats; we'll try common patterns
    
    # Pattern 1: <tool_call> ... </tool_call> blocks
    pattern1 = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
    
    # Pattern 2: JSON objects with "name" and "arguments" keys that look like tool calls
    # This is a fallback for cases where tool calls are embedded differently
    pattern2 = re.compile(r'\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', re.DOTALL)
    
    # Try pattern 1 first
    matches1 = pattern1.findall(raw_template)
    for match in matches1:
        try:
            tool_call = json.loads(match)
            if 'name' in tool_call and 'arguments' in tool_call:
                tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    
    # If no matches from pattern 1, try pattern 2
    if not tool_calls:
        matches2 = pattern2.findall(raw_template)
        for match in matches2:
            try:
                tool_call = json.loads(match)
                if 'name' in tool_call and 'arguments' in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
    
    return tool_calls

@app.get("/health")
async def health_check():
  return {
      "status": "ready" if inference_service.llm is not None else "loading",
      "model": cfg.get("model_path")
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
from harness import CodingHarness
ch = CodingHarness()
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
    reply_string = reply['choices'][0]['text']
    tools = extract_tool_calls(reply_string) 
    print(f"{tools}")
    for tool in tools:
        print(f"{tool}")
    # content = reply['choices'][0]['text'];
    return reply
