"""Runtime Gemini wrapper (programmatic LLM calls — NOT Claude sub-agents).

Every node in the deployed agent calls Gemini through here. Provides a tolerant
JSON helper because gemini-3.5-flash returns content as a list of blocks and may
wrap JSON in prose/fences. LangSmith tracing is enabled via env.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

from src.ingestion.config import ROOT

load_dotenv(dotenv_path=ROOT / ".env")

MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash")

# LangSmith tracing: mirror the new env var onto the names LangChain checks.
if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if os.environ.get("LANGSMITH_PROJECT"):
        os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=temperature,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or "")
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return str(content)


def render(template: str, **kwargs) -> str:
    """Fill {name} placeholders without str.format (templates contain literal JSON braces)."""
    out = template
    for key, val in kwargs.items():
        out = out.replace("{" + key + "}", str(val))
    return out


def call_text(system: str, user: str, temperature: float = 0.0) -> str:
    msg = get_llm(temperature).invoke([("system", system), ("human", user)])
    return _content_to_text(msg.content).strip()


def call_json(system: str, user: str, temperature: float = 0.0, retries: int = 1):
    """Call Gemini and parse a single JSON object; returns None if unparseable."""
    suffix = "\n\nReturn ONLY a single valid JSON object. No markdown fences, no commentary."
    u = user + suffix
    for _ in range(retries + 1):
        parsed = _extract_json(call_text(system, u, temperature))
        if parsed is not None:
            return parsed
        u = user + "\n\nYour previous reply was not valid JSON. Output ONLY one valid JSON object."
    return None


def _extract_json(raw: str):
    if not raw:
        return None
    s = re.sub(r"^```[a-zA-Z]*", "", raw.strip()).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None
