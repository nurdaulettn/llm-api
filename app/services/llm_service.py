import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from google import genai


from app.schemas.llm_schema import LLMRequest, LLMResponse, Usage

logger = logging.getLogger(__name__)
load_dotenv()

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment")
            raise RuntimeError("GEMINI_API_KEY not found. Check your .env and load_dotenv().")

        logger.info("Initializing Gemini client", extra={"@timestamp": datetime.utcnow().isoformat() + "Z"})
        _client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized successfully", extra={"@timestamp": datetime.utcnow().isoformat() + "Z"})
    return _client



async def ask_llm(request: LLMRequest) -> LLMResponse:
    client = _get_client()
    primary_model = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash")
    fallback_models_raw = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-1.5-flash")

    # primary_model = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-1.5-flash")
    # fallback_models_raw = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash")
    fallback_models = [m.strip() for m in fallback_models_raw.split(",") if m.strip()]

    models_to_try = [primary_model] + [m for m in fallback_models if m != primary_model]

    logger.info("LLM request started", extra={
        "@timestamp": datetime.utcnow().isoformat() + "Z",
        "models_to_try": models_to_try,
    })

    last_error: Exception | None = None

    for attempt, model_name in enumerate(models_to_try, start=1):
        start = time.perf_counter()

        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=request.question,
            )

            latency_ms = int((time.perf_counter() - start) * 1000)

            answer = getattr(response, "text", "")
            usage_metadata = getattr(response, "usage_metadata", None)

            if usage_metadata:
                usage = Usage(
                    prompt_tokens=getattr(usage_metadata, "prompt_token_count", 0),
                    completion_tokens=getattr(usage_metadata, "candidates_token_count", 0),
                    total_tokens=getattr(usage_metadata, "total_token_count", 0),
                )
            else:
                usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

            logger.info("LLM request finished", extra={
                "@timestamp": datetime.utcnow().isoformat() + "Z",
                "model": model_name,
                "attempt": attempt,
                "latency_ms": latency_ms,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "used_fallback": (model_name != primary_model),
            })

            return LLMResponse(
                answer=answer,
                model=model_name,
                usage=usage,
                latency_ms=latency_ms,
            )

        except Exception as e:
            last_error = e
            logger.warning("Gemini model failed, trying next (if any)", extra={
                "@timestamp": datetime.utcnow().isoformat() + "Z",
                "failed_model": model_name,
                "attempt": attempt,
                "error": str(e),
            })

    # если все модели упали
    logger.exception("All Gemini models failed", extra={
        "@timestamp": datetime.utcnow().isoformat() + "Z",
        "models_tried": models_to_try,
        "last_error": str(last_error) if last_error else "unknown",
    })
    raise RuntimeError(f"Gemini API error: {last_error}") from last_error