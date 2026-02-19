import os
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from starlette import status

from app.logging.logging_config import setup_logging
from app.schemas.llm_schema import LLMRequest, LLMResponse, HealthResponse
from app.services.llm_service import ask_llm
router = APIRouter(tags=["LLM"])

logger = logging.getLogger(__name__)




@router.post(
    "/generate",
    response_model=LLMResponse,
    summary="Generate answer using LLM",
)
async def generate(request: LLMRequest):
    try:
        return await ask_llm(request)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("LLM generation failed", extra={
            "@timestamp": datetime.utcnow().isoformat() + "Z"
        })
        setup_logging()
        raise HTTPException(
            status_code=500,
            detail="LLM service temporarily unavailable"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check"
)
async def health_check():
    api_key_present = bool(os.getenv("GEMINI_API_KEY"))

    return HealthResponse(
        status="healthy" if api_key_present else "degraded",
        service="LLM API",
        api_key_configured=api_key_present
    )