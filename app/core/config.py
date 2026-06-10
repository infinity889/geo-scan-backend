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
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv(
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1",
    )
    groq_vision_model: str = os.getenv(
        "GROQ_VISION_MODEL",
        "llama-3.2-11b-vision-preview",
    )
    groq_llm_model: str = os.getenv(
        "GROQ_LLM_MODEL",
        "llama-3.1-8b-instant",
    )
    groq_timeout_seconds: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))

    database_url: str = os.getenv("DATABASE_URL", "")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

    @property
    def database_enabled(self) -> bool:
        return bool(self.database_url)

    @property
    def groq_enabled(self) -> bool:
        return bool(self.groq_api_key)

    @cached_property
    def cors_origins(self) -> list[str]:
        raw = os.getenv(
            "BACKEND_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        )
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


settings = Settings()
