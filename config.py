from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    model_path: str = "./cnn_model.keras"
    allowed_origins: List[str] = ["*"]
    rate_limit_predict: str = "10/minute"
    rate_limit_gemini: str = "5/minute"
    max_file_size_mb: int = 10
    allowed_content_types: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
    ]
    debug: bool = False

    model_config = {"env_prefix": "POTATO_", "env_file": ".env", "protected_namespaces": ("settings_",)}


settings = Settings()
