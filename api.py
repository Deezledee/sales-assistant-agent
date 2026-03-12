from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from agent import build_sales_agent


class QuestionRequest(BaseModel):
    question: str


app = FastAPI(title="Sales Assistant Agent API")


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
    response = get_agent_executor().invoke({"input": request.question})
    return {"answer": response.get("output", "No answer generated.")}
