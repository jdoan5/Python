# settings from env vars

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Hello-Prod-FastAPI"
    environment: str = "local"
    debug: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()