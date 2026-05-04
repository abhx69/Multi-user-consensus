"""
Configuration management for Gaprio Agent Bot.

Uses Pydantic Settings for type-safe configuration with environment variable loading.
All settings can be overridden via environment variables or a .env file.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Settings are organized by component:
    - Slack: Bot and app tokens for Slack integration
    - LLM: Provider configuration (Ollama or OpenAI)
    - RAG: Vector store and indexing configuration
    - Memory: File-based memory settings
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # -------------------------------------------------------------------------
    # Slack Configuration
    # -------------------------------------------------------------------------
    slack_bot_token: str = Field(
        default="",
        description="Slack Bot OAuth Token (xoxb-...)"
    )
    slack_app_token: str = Field(
        default="",
        description="Slack App-Level Token for Socket Mode (xapp-...)"
    )
    slack_signing_secret: str = Field(
        default="",
        description="Slack Signing Secret for request verification"
    )
    
    # -------------------------------------------------------------------------
    # LLM Provider Configuration
    # -------------------------------------------------------------------------
    llm_provider: Literal["ollama", "openai", "openrouter", "bedrock"] = Field(
        default="ollama",
        description="LLM provider to use: 'ollama' (local), 'openai', 'openrouter', or 'bedrock'"
    )
    
    # Bedrock settings
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-haiku-20240307-v1:0",
        description="Bedrock model ID"
    )
    aws_access_key_id: str = Field(
        default="",
        description="AWS Access Key ID"
    )
    aws_secret_access_key: str = Field(
        default="",
        description="AWS Secret Access Key"
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS Region"
    )
    
    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    ollama_model: str = Field(
        default="llama3:instruct",
        description="Ollama model to use for inference"
    )
    
    # OpenAI settings
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4",
        description="OpenAI model to use"
    )
    
    # OpenRouter settings
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key"
    )
    openrouter_model: str = Field(
        default="openai/gpt-oss-120b:free",
        description="OpenRouter model to use"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )

    
    # -------------------------------------------------------------------------
    # Asana Configuration
    # -------------------------------------------------------------------------
    asana_client_id: str = Field(
        default="",
        description="Asana OAuth Client ID"
    )
    asana_client_secret: str = Field(
        default="",
        description="Asana OAuth Client Secret"
    )
    asana_access_token: str = Field(
        default="",
        description="Asana Access Token (from OAuth flow)"
    )
    asana_refresh_token: str = Field(
        default="",
        description="Asana Refresh Token (for token refresh)"
    )
    asana_default_workspace: str = Field(
        default="",
        description="Default Asana Workspace GID"
    )
    
    # -------------------------------------------------------------------------
    # Google Workspace Configuration
    # -------------------------------------------------------------------------
    google_client_id: str = Field(
        default="",
        description="Google OAuth Client ID"
    )
    google_client_secret: str = Field(
        default="",
        description="Google OAuth Client Secret"
    )
    google_refresh_token: str = Field(
        default="",
        description="Google OAuth Refresh Token"
    )
    google_redirect_uri: str = Field(
        default="http://localhost:8000/api/integrations/google/callback",
        description="Google OAuth Redirect URI"
    )
    
    # -------------------------------------------------------------------------
    # RAG Configuration
    # -------------------------------------------------------------------------
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        description="Directory for ChromaDB persistence"
    )
    rag_index_message_count: int = Field(
        default=200,
        description="Number of past messages to index per channel"
    )
    rag_index_frequency_hours: int = Field(
        default=2,
        description="How often to re-index channel messages (in hours)"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings"
    )
    
    # -------------------------------------------------------------------------
    # Memory Configuration
    # -------------------------------------------------------------------------
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory for memory files"
    )
    max_conversation_history: int = Field(
        default=20,
        description="Maximum messages to keep in conversation context"
    )
    
    # -------------------------------------------------------------------------
    # MySQL Database (shared with Express backend for OAuth tokens)
    # -------------------------------------------------------------------------
    db_host: str = Field(
        default="localhost",
        description="MySQL host (shared with gaprio-backend)"
    )
    db_port: int = Field(
        default=3306,
        description="MySQL port"
    )
    db_user: str = Field(
        default="root",
        description="MySQL user"
    )
    db_password: str = Field(
        default="",
        description="MySQL password"
    )
    db_name: str = Field(
        default="gapriomanagement",
        description="MySQL database name (must match Express backend)"
    )
    
    # -------------------------------------------------------------------------
    # External MCP Configuration
    # -------------------------------------------------------------------------
    external_mcp_servers: dict[str, str] = Field(
        default_factory=dict,
        description="Map of external MCP servers. Key=name, Value=command (e.g. 'npx -y ...')"
    )
    
    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "memory").mkdir(exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance - import this in other modules
settings = Settings()
