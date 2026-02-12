from pydantic import BaseModel


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