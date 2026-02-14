import os
import time
from typing import Any, Coroutine

from dotenv import load_dotenv
from google import genai

from app.schemas.llm_schema import LLMRequest, LLMResponse, Usage

load_dotenv()

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """
    Создаёт клиента один раз и переиспользует.
    Не падаем на import-этапе, а проверяем ключ при первом использовании.
    """
    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found. Check your .env and load_dotenv().")

        _client = genai.Client(api_key=api_key)

    return _client


async def ask_llm(request: LLMRequest) -> LLMResponse:
    client = _get_client()

    start = time.perf_counter()
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.question,
        )
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}") from e
    finally:
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

    return LLMResponse(
        answer=answer,
        model="gemini-2.5-flash",
        usage=usage,
        latency_ms=latency_ms,
    )