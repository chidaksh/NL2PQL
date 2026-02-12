from __future__ import annotations
import os
from langchain_openai import ChatOpenAI

def get_llm(temperature: float = 0.3, model_name='gpt-4o-mini'):
    if os.environ.get("OPENAI_API_KEY"):
        return ChatOpenAI(model=model_name, temperature=temperature)
    raise ValueError("OPENAI_API_KEY not set. Add it to .env or export it.")