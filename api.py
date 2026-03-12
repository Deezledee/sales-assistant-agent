from functools import lru_cache
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import build_sales_agent


class QuestionRequest(BaseModel):
    question: str


app = FastAPI(title="Sales Assistant Agent API")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_agent_executor():
    return build_sales_agent(verbose=False)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Sales Assistant Agent API is running.",
        "health": "/health",
        "ask": "/ask",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QuestionRequest) -> dict[str, str]:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        response = get_agent_executor().invoke({"input": request.question})
        return {"answer": response.get("output", "No answer generated.")}
    except Exception as error:
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {error}") from error
