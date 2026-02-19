from fastapi import FastAPI

from app.logging.logging_config import setup_logging
from app.routers import llm_router

app = FastAPI()

setup_logging()

app.include_router(llm_router.router)
