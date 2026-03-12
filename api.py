from functools import lru_cache
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import build_sales_agent


class QuestionRequest(BaseModel):
    question: str


app = FastAPI(title="Sales Assistant Agent API")
logger = logging.getLogger(__name__)

CHAT_HTML = """
<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Sales Assistant Chat</title>
        <style>
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                color: #111827;
            }
            .container {
                max-width: 800px;
                margin: 24px auto;
                padding: 16px;
            }
            .card {
                background: #ffffff;
                border-radius: 12px;
                box-shadow: 0 1px 8px rgba(0, 0, 0, 0.08);
                overflow: hidden;
            }
            .header {
                padding: 16px;
                border-bottom: 1px solid #e5e7eb;
                font-size: 18px;
                font-weight: 700;
            }
            .messages {
                height: 460px;
                overflow-y: auto;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                background: #fbfbfc;
            }
            .msg {
                padding: 10px 12px;
                border-radius: 10px;
                max-width: 85%;
                line-height: 1.4;
                white-space: pre-wrap;
            }
            .user {
                align-self: flex-end;
                background: #2563eb;
                color: #fff;
            }
            .agent {
                align-self: flex-start;
                background: #e5e7eb;
            }
            .footer {
                padding: 12px;
                border-top: 1px solid #e5e7eb;
                display: flex;
                gap: 8px;
            }
            input {
                flex: 1;
                padding: 10px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                font-size: 14px;
            }
            button {
                padding: 10px 14px;
                border: none;
                border-radius: 8px;
                background: #111827;
                color: #fff;
                cursor: pointer;
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .hint {
                font-size: 12px;
                color: #6b7280;
                margin-top: 8px;
            }
        </style>
    </head>
    <body>
        <div class=\"container\">
            <div class=\"card\">
                <div class=\"header\">Sales Assistant Agent</div>
                <div id=\"messages\" class=\"messages\"></div>
                <div class=\"footer\">
                    <input id=\"question\" type=\"text\" placeholder=\"Ask about customer plans, pricing, or policy...\" />
                    <button id=\"send\">Send</button>
                </div>
            </div>
            <div class=\"hint\">Examples: \"What plan does John Smith have?\", \"How much is the Growth plan?\"</div>
        </div>

        <script>
            const messages = document.getElementById('messages');
            const input = document.getElementById('question');
            const sendButton = document.getElementById('send');

            function addMessage(text, role) {
                const div = document.createElement('div');
                div.className = `msg ${role}`;
                div.textContent = text;
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }

            async function sendQuestion() {
                const question = input.value.trim();
                if (!question) return;

                addMessage(question, 'user');
                input.value = '';
                sendButton.disabled = true;

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question })
                    });

                    const data = await response.json();
                    if (!response.ok) {
                        addMessage(`Error: ${data.detail || 'Request failed.'}`, 'agent');
                    } else {
                        addMessage(data.answer || 'No answer generated.', 'agent');
                    }
                } catch (error) {
                    addMessage('Error: Could not reach the API.', 'agent');
                } finally {
                    sendButton.disabled = false;
                    input.focus();
                }
            }

            sendButton.addEventListener('click', sendQuestion);
            input.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') sendQuestion();
            });

            addMessage('Hi! I can answer CRM, pricing, and policy questions.', 'agent');
            input.focus();
        </script>
    </body>
</html>
"""


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
        "chat": "/chat",
    }


@app.get("/chat", response_class=HTMLResponse)
def chat() -> str:
    return CHAT_HTML


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
