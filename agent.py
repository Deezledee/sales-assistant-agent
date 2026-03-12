from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

CRM_DATA = {
    "john smith": {
        "company": "Smith & Co",
        "subscription": "Growth",
        "last_invoice": "2026-02-28 (€49)",
    },
    "acme ltd": {
        "company": "Acme Ltd",
        "subscription": "Scale",
        "last_invoice": "2026-02-28 (€99)",
    },
    "maria garcia": {
        "company": "Garcia Retail",
        "subscription": "Starter",
        "last_invoice": "2026-02-28 (€19)",
    },
}

PRICING_DATA = {
    "starter": "€19/month",
    "growth": "€49/month",
    "scale": "€99/month",
}


def _normalize_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", value.lower())
    return " ".join(normalized.split())


def _find_customer_key(raw_input: str) -> str | None:
    direct_key = raw_input.strip().lower()
    if direct_key in CRM_DATA:
        return direct_key

    normalized_input = _normalize_text(raw_input)
    for customer_key in CRM_DATA:
        if _normalize_text(customer_key) in normalized_input:
            return customer_key

    return None


def _find_plan_key(raw_input: str) -> str | None:
    direct_key = raw_input.strip().lower()
    if direct_key in PRICING_DATA:
        return direct_key

    normalized_input = _normalize_text(raw_input)
    for plan_key in PRICING_DATA:
        if _normalize_text(plan_key) in normalized_input:
            return plan_key

    return None


class SimpleRAGRetriever:
    def __init__(self, kb_path: Path, chunk_size: int = 450, chunk_overlap: int = 50) -> None:
        self.kb_path = kb_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.chunks = self._load_chunks()
        if not self.chunks:
            self.vectors = np.empty((0, 0), dtype=float)
        else:
            self.vectors = np.array(self.embeddings.embed_documents(self.chunks), dtype=float)

    def _load_chunks(self) -> list[str]:
        if not self.kb_path.exists():
            return []

        text = self.kb_path.read_text(encoding="utf-8")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)

    def search(self, query: str, top_k: int = 3) -> str:
        if not self.chunks or self.vectors.size == 0:
            return "Knowledge base is empty."

        query_vec = np.array(self.embeddings.embed_query(query), dtype=float)

        doc_norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denominator = np.maximum(doc_norms * query_norm, 1e-10)
        similarities = (self.vectors @ query_vec) / denominator

        k = min(top_k, len(self.chunks))
        top_indices = similarities.argsort()[-k:][::-1]
        results = [self.chunks[index] for index in top_indices]

        return "\n\n".join(results)


def get_customer(customer_name: str) -> str:
    key = _find_customer_key(customer_name)
    customer = CRM_DATA.get(key) if key else None
    if customer is None:
        available_customers = ", ".join(name.title() for name in CRM_DATA)
        return (
            f"No customer found for '{customer_name}'. "
            f"Available customers: {available_customers}."
        )

    return (
        f"Company: {customer['company']}; "
        f"Subscription: {customer['subscription']}; "
        f"Last invoice: {customer['last_invoice']}"
    )


def get_pricing(plan_name: str) -> str:
    key = _find_plan_key(plan_name)
    price = PRICING_DATA.get(key) if key else None
    if price is None:
        available = ", ".join(plan.title() for plan in PRICING_DATA)
        return f"Unknown plan '{plan_name}'. Available plans: {available}."

    return f"The {key.title()} plan costs {price}."


def build_sales_agent(verbose: bool = True):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    kb_path = Path(__file__).parent / "data" / "knowledge_base.md"
    rag_retriever = SimpleRAGRetriever(kb_path=kb_path)

    def search_knowledge_base(query: str) -> str:
        return rag_retriever.search(query)

    tools = [
        Tool(
            name="Customer Lookup",
            func=get_customer,
            description=(
                "Use this for questions about a customer's company, "
                "subscription plan, or last invoice. Input should be a customer name."
            ),
        ),
        Tool(
            name="Pricing Lookup",
            func=get_pricing,
            description="Use this for plan pricing questions. Input should be a plan name.",
        ),
        Tool(
            name="Knowledge Base Retrieval",
            func=search_knowledge_base,
            description=(
                "Use this for policies, FAQs, or product documentation. "
                "Input should be the user's question."
            ),
        ),
    ]

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        handle_parsing_errors=True,
    )


def ask_question(question: str, verbose: bool = False) -> str:
    agent_executor = build_sales_agent(verbose=verbose)
    response: dict[str, Any] = agent_executor.invoke({"input": question})
    return str(response.get("output", "No answer generated."))
