import os

from fastapi import APIRouter
from starlette import status

from app.schemas.llm_schema import LLMRequest, LLMResponse
from app.services.llm_service import ask_llm

router = APIRouter(tags=["LLM"])


@router.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest):
    return await ask_llm(request)


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))

    return {
        "status": "healthy" if api_key_present else "degraded",
        "service": "LLM API",
        "api_key_configured": api_key_present
    }