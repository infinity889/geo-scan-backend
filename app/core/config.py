import os
from functools import cached_property
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    app_name: str = os.getenv("APP_NAME", "Geo Scan Backend")
    app_env: str = os.getenv("APP_ENV", "development")
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    openrouter_referer: str = os.getenv("OPENROUTER_HTTP_REFERER", "")
    openrouter_title: str = os.getenv("OPENROUTER_APP_TITLE", "Geo Scan KMG")
    openrouter_ocr_model: str = os.getenv(
        "OPENROUTER_OCR_MODEL",
        "deepseek/deepseek-ocr-2",
    )
    openrouter_llm_model: str = os.getenv(
        "OPENROUTER_LLM_MODEL",
        "deepseek/deepseek-v4-flash",
    )
    openrouter_embedding_model: str = os.getenv(
        "OPENROUTER_EMBEDDING_MODEL",
        "baai/bge-m3",
    )
    openrouter_timeout_seconds: float = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "45"))

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.openrouter_api_key)

    @cached_property
    def cors_origins(self) -> list[str]:
        raw = os.getenv(
            "BACKEND_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        )
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


settings = Settings()
