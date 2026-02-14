from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    service: str
    api_key_configured: bool


class LLMRequest(BaseModel):
    question: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class LLMResponse(BaseModel):
    answer: str
    model: str
    usage: Usage
    latency_ms: int
