import os
from typing import List
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "GST Reconciliation API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)
    
    # API settings
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["*"]
    
    # File storage settings
    UPLOAD_DIR: str = Field(default="./data/uploads")
    REPORTS_DIR: str = Field(default="./data/reports")
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB
    
    # Redis settings
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_TTL: int = 3600  # Cache TTL in seconds
    
    # ML Model settings
    ML_MODEL_PATH: str = Field(default="./app/ml/models/reconciliation_model.joblib")
    EMBEDDINGS_MODEL: str = "paraphrase-distilroberta-base-v1"
    SIMILARITY_THRESHOLD: float = 0.85
    
    # Fuzzy matching settings
    FUZZY_THRESHOLD: int = 85
    
    # Reconciliation settings
    DATE_FORMAT: str = "%d-%m-%Y"
    CSV_DATE_FORMAT: str = "%d-%b-%Y"
    MAX_WORKERS: int = 4  # Number of parallel workers for processing
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REPORTS_DIR, exist_ok=True)