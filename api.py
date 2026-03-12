from functools import lru_cache
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import build_sales_agent, get_customer, get_pricing


class QuestionRequest(BaseModel):
    question: str


app = FastAPI(title="Sales Assistant Agent API")
logger = logging.getLogger(__name__)

KNOWN_CUSTOMERS = ("john smith", "acme ltd", "maria garcia")
KNOWN_PLANS = ("starter", "growth", "scale")


def _get_deploy_version() -> str:
    commit = os.getenv("RENDER_GIT_COMMIT")
    if not commit:
        return "local"
    return commit[:7]


def _try_direct_tool_answer(question: str) -> str | None:
    lowered = question.lower()

    if any(customer in lowered for customer in KNOWN_CUSTOMERS):
        if any(keyword in lowered for keyword in ("plan", "subscription", "invoice", "company", "customer")):
            return get_customer(question)

    if any(plan in lowered for plan in KNOWN_PLANS):
        if any(keyword in lowered for keyword in ("price", "pricing", "cost", "plan", "month", "monthly")):
            return get_pricing(question)

    return None

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
                background: #050b18;
                color: #e5edff;
            }
            .container {
                max-width: 800px;
                margin: 24px auto;
                padding: 16px;
            }
            .card {
                background: #0b1220;
                border-radius: 12px;
                border: 1px solid #1b2b49;
                box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
                overflow: hidden;
            }
            .header {
                padding: 16px;
                border-bottom: 1px solid #1b2b49;
                font-size: 18px;
                font-weight: 700;
                color: #f8fbff;
            }
            .messages {
                height: 460px;
                overflow-y: auto;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                background: #0a1326;
            }
            .prompt-list {
                padding: 12px 12px 0;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                border-bottom: 1px solid #1b2b49;
            }
            .prompt-btn {
                padding: 6px 10px;
                border: 1px solid #2a4a82;
                border-radius: 999px;
                background: #0f1f3d;
                color: #dbeafe;
                font-size: 12px;
                cursor: pointer;
            }
            .prompt-btn:hover {
                background: #1c3568;
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
                background: #1d4ed8;
                color: #fff;
            }
            .agent {
                align-self: flex-start;
                background: #16243d;
                border: 1px solid #27406b;
                color: #eff6ff;
            }
            .footer {
                padding: 12px;
                border-top: 1px solid #1b2b49;
                display: flex;
                gap: 8px;
            }
            input {
                flex: 1;
                padding: 10px;
                border: 1px solid #27406b;
                border-radius: 8px;
                font-size: 14px;
                background: #0f1a31;
                color: #e5edff;
            }
            input::placeholder {
                color: #9ab0d6;
            }
            button {
                padding: 10px 14px;
                border: none;
                border-radius: 8px;
                background: #2563eb;
                color: #fff;
                cursor: pointer;
                font-weight: 600;
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .hint {
                font-size: 12px;
                color: #9ab0d6;
                margin-top: 8px;
            }
            .version {
                margin-top: 8px;
                font-size: 11px;
                color: #7d97c4;
            }
        </style>
    </head>
    <body>
        <div class=\"container\">
            <div class=\"card\">
                <div class=\"header\">Sales Assistant Agent</div>
                <div class="prompt-list">
                    <button class="prompt-btn" data-prompt="What plan does John Smith have?">John Smith plan</button>
                    <button class="prompt-btn" data-prompt="How much is the Growth plan?">Growth pricing</button>
                    <button class="prompt-btn" data-prompt="What is your upgrade policy?">Upgrade policy</button>
                    <button class="prompt-btn" data-prompt="Summarize billing policy in 2 bullets.">Billing summary</button>
                </div>
                <div id=\"messages\" class=\"messages\"></div>
                <div class=\"footer\">
                    <input id=\"question\" type=\"text\" placeholder=\"Ask about customer plans, pricing, or policy...\" />
                    <button id=\"send\">Send</button>
                </div>
            </div>
            <div class=\"hint\">Examples: \"What plan does John Smith have?\", \"How much is the Growth plan?\"</div>
            <div class=\"version\">Deploy version: __VERSION__</div>
        </div>

        <script>
            const messages = document.getElementById('messages');
            const input = document.getElementById('question');
            const sendButton = document.getElementById('send');
            const promptButtons = document.querySelectorAll('.prompt-btn');

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
            promptButtons.forEach((button) => {
                button.addEventListener('click', () => {
                    input.value = button.dataset.prompt || '';
                    input.focus();
                });
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
        "version": _get_deploy_version(),
        "health": "/health",
        "ask": "/ask",
        "docs": "/docs",
        "chat": "/chat",
    }


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": _get_deploy_version()}


@app.get("/chat", response_class=HTMLResponse)
def chat() -> str:
    return CHAT_HTML.replace("__VERSION__", _get_deploy_version())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QuestionRequest) -> dict[str, str]:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        direct_answer = _try_direct_tool_answer(request.question)
        if direct_answer is not None:
            return {"answer": direct_answer}

        response = get_agent_executor().invoke({"input": request.question})
        return {"answer": response.get("output", "No answer generated.")}
    except Exception as error:
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {error}") from error
