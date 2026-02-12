import os
import google.generativeai as genai
from dotenv import load_dotenv
from app.schemas.llm_schema import LLMRequest, LLMResponse, Usage

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=API_KEY)


async def ask_llm(request: LLMRequest) -> LLMResponse:
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(request.question)

        answer = response.text

        usage_metadata = getattr(response, 'usage_metadata', None)

        if usage_metadata:
            usage = Usage(
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
                total_tokens=usage_metadata.total_token_count
            )
        else:
            usage = Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )

        return LLMResponse(
            answer=answer,
            model="gemini-2.5-flash",
            usage=usage
        )

    except Exception as e:
        raise Exception(f"Gemini API error: {str(e)}")

