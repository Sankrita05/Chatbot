from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # === Project Metadata === #
    PROJECT_NAME: str = "Document Upload and Vectorizer API"
    API_VERSION: str = "v1"

    # === File Directories === #
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
    MODEL_DIR: str = os.path.join(BASE_DIR, "models")
    CHROMA_DB_DIR: str = os.path.join(BASE_DIR, "chroma_db")

    # === Model Settings === #
    MODEL_NAME: str = "all-MiniLM-L6-v2"  # You can change this to another SentenceTransformer model

    # === ChromaDB Settings === #
    COLLECTION_NAME: str = "document_embeddings"

    # === Security === #
    SECRET_KEY: str = "your_super_secret_key_here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # DeepSeek API settings
    DEEPSEEK_API_KEY: str
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"

    # MongoDB
    MONGODB_URI: str
    MONGODB_DB_NAME: str
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
