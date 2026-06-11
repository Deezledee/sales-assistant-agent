# Sales Assistant Agent

An AI-powered sales assistant built with **Python**, **LangChain**, and the **OpenAI API**. The agent uses a ReAct tool-calling architecture to dynamically answer customer questions by routing across multiple tools and a RAG-based knowledge base.

## Features

- **CRM lookup** — retrieve customer information by name
- **Pricing lookup** — look up plan pricing on demand
- **Knowledge base (RAG)** — semantic search over internal docs using OpenAI embeddings
- **ReAct agent** — dynamic tool selection via LangChain
- **CLI mode** — run locally in the terminal
- **API mode** — FastAPI endpoint for remote access

## Tech stack

| Layer | Library |
|---|---|
| LLM | OpenAI |
| Agent framework | LangChain |
| Embeddings | OpenAI Embeddings |
| API server | FastAPI + Uvicorn |

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your API key.

3. Run the CLI:

   ```bash
   python main.py
   ```

   Or start the API server:

   ```bash
   uvicorn api:app --reload
   ```

## Deployment

A `render.yaml` is included for one-click deployment to [Render](https://render.com).
