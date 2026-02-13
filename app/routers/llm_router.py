import os

from fastapi import APIRouter
from starlette import status

from app.schemas.llm_schema import LLMRequest, LLMResponse
from app.services.llm_service import ask_llm
from fastapi import HTTPException
router = APIRouter(tags=["LLM"])


import logging

logger = logging.getLogger(__name__)

@router.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest):
    try:
        response = await ask_llm(request)
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("LLM generation failed")
        raise HTTPException(
            status_code=500,
            detail="LLM service temporarily unavailable"
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    api_key_present = bool(os.getenv("GEMINI_API_KEY"))

    return {
        "status": "healthy" if api_key_present else "degraded",
        "service": "LLM API",
        "api_key_configured": api_key_present
    }