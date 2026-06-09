import hashlib
import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter is not configured or returns an invalid response."""


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    model: str
    raw: dict[str, Any]


class OpenRouterClient:
    def __init__(self) -> None:
        self.base_url = settings.openrouter_base_url.rstrip("/")

    @property
    def configured(self) -> bool:
        return settings.openrouter_enabled

    def _headers(self) -> dict[str, str]:
        if not settings.openrouter_api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if settings.openrouter_referer:
            headers["HTTP-Referer"] = settings.openrouter_referer
        if settings.openrouter_title:
            headers["X-OpenRouter-Title"] = settings.openrouter_title
        return headers

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        response_format: dict[str, str] | None = None,
    ) -> ChatCompletionResult:
        model_name = model or settings.openrouter_llm_model
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        data = await self._post_json("/chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError("OpenRouter returned a malformed chat response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise OpenRouterError("OpenRouter returned an empty chat response.")
        return ChatCompletionResult(content=content, model=model_name, raw=data)

    async def embeddings(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        model_name = model or settings.openrouter_embedding_model
        payload = {"model": model_name, "input": texts}
        data = await self._post_json("/embeddings", payload)
        try:
            return [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError) as exc:
            raise OpenRouterError("OpenRouter returned a malformed embedding response.") from exc

    async def ocr_image(
        self,
        image_data_url: str,
        *,
        prompt: str | None = None,
        model: str | None = None,
    ) -> ChatCompletionResult:
        instruction = prompt or (
            "Parse this geological page. Preserve tables as Markdown, formulas as "
            "LaTeX, captions, page structure, and Russian geological terminology."
        )
        return await self.chat_completion(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            model=model or settings.openrouter_ocr_model,
            temperature=0,
            max_tokens=2500,
        )

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=settings.openrouter_timeout_seconds) as client:
            response = await client.post(url, headers=self._headers(), json=payload)

        if response.status_code >= 400:
            raise OpenRouterError(
                f"OpenRouter request failed: {response.status_code} {response.text[:500]}"
            )
        return response.json()


def deterministic_embedding(text: str, dimensions: int = 32) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for idx in range(dimensions):
        byte = digest[idx % len(digest)]
        values.append(round((byte / 127.5) - 1.0, 6))
    return values


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)


openrouter_client = OpenRouterClient()
