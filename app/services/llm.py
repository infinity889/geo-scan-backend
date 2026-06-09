import hashlib
import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings


class LLMError(RuntimeError):
    """Raised when the LLM provider returns an error."""


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    model: str
    raw: dict[str, Any]


class GroqClient:
    def __init__(self) -> None:
        self.base_url = settings.groq_base_url.rstrip("/")

    @property
    def configured(self) -> bool:
        return settings.groq_enabled

    def _headers(self) -> dict[str, str]:
        if not settings.groq_api_key:
            raise LLMError("GROQ_API_KEY is not configured.")

        return {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        response_format: dict[str, str] | None = None,
    ) -> ChatCompletionResult:
        model_name = model or settings.groq_llm_model
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
            raise LLMError("Groq returned a malformed chat response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMError("Groq returned an empty chat response.")
        return ChatCompletionResult(content=content, model=model_name, raw=data)

    async def ocr_image(
        self,
        image_data_url: str,
        *,
        prompt: str | None = None,
        model: str | None = None,
    ) -> ChatCompletionResult:
        """Use Groq's vision models for image analysis."""
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
            model=model or settings.groq_vision_model,
            temperature=0,
            max_tokens=2500,
        )

    async def embeddings(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Groq doesn't provide embeddings. Falling back to deterministic hashing."""
        return [deterministic_embedding(t) for t in texts]

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=settings.groq_timeout_seconds) as client:
            response = await client.post(url, headers=self._headers(), json=payload)

        if response.status_code >= 400:
            raise LLMError(
                f"Groq request failed: {response.status_code} {response.text[:500]}"
            )
        return response.json()


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        # Multilingual E5-Large model (1024 dims) as requested in requirements
        _embedder = TextEmbedding(model_name="intfloat/multilingual-e5-large")
    return _embedder


def deterministic_embedding(text: str, dimensions: int | None = None) -> list[float]:
    """Generates real semantic embeddings using BGE-M3."""
    embedder = get_embedder()
    # fastembed returns a generator
    embeddings = list(embedder.embed([text]))
    vector = embeddings[0].tolist()

    target_dim = dimensions or settings.embedding_dimensions
    if len(vector) != target_dim:
        # Resize if necessary (BGE-M3 is 1024 by default)
        if len(vector) > target_dim:
            return vector[:target_dim]
        else:
            return vector + [0.0] * (target_dim - len(vector))
    return vector


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)


llm_client = GroqClient()
# Add alias for backward compatibility during refactoring
openrouter_client = llm_client
