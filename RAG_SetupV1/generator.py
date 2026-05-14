"""
generator.py  —  V1
===================
Direct Ollama call via requests.post.
"""

from __future__ import annotations

import requests


PROMPT_TEMPLATE = """\
You are a driver-safety assistant. Answer in 1-2 short sentences a
driver can hear in traffic. Ground your answer in the context below.

Context:
{context}

Question: {query}

Answer:"""


class OllamaGenerator:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.1",
        temperature: float = 0.2,
        timeout_sec: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def generate(self, query: str, chunks: list[str]) -> str:
        prompt = PROMPT_TEMPLATE.format(
            context="\n".join(f"- {c}" for c in chunks),
            query=query,
        )
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": 200},
            },
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()
