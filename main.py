from fastapi import FastAPI
from app.routers import llm_router

app = FastAPI()
app.include_router(llm_router.router)
