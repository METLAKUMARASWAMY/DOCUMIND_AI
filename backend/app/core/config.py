from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", alias="APP_ENV")
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    mongodb_uri: str = Field(default="mongodb://mongodb:27017/documind", alias="MONGODB_URI")
    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    minio_endpoint: str = Field(default="minio:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str | None = Field(default=None, alias="MINIO_ACCESS_KEY")
    minio_secret_key: str | None = Field(default=None, alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="documind-documents", alias="MINIO_BUCKET")
    chroma_host: str = Field(default="chroma", alias="CHROMA_HOST")
    neo4j_uri: str = Field(default="bolt://neo4j:7687", alias="NEO4J_URI")
    neo4j_user: str | None = Field(default=None, alias="NEO4J_USER")
    neo4j_password: str | None = Field(default=None, alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    documind_dataset_path: str = Field(
        default="dataset/documind_ai_synthetic_dataset",
        alias="DOCUMIND_DATASET_PATH",
    )
    rvl_cdip_path: str | None = Field(default=None, alias="RVL_CDIP_PATH")
    auth_secret: str | None = Field(default=None, alias="DOCUMIND_AUTH_SECRET")
    auth_token_ttl_seconds: int = Field(default=15 * 60, alias="DOCUMIND_AUTH_TOKEN_TTL_SECONDS")
    auth_refresh_ttl_seconds: int = Field(
        default=7 * 24 * 60 * 60,
        alias="DOCUMIND_AUTH_REFRESH_TTL_SECONDS",
    )
    auth_state_dir: str = Field(default="var/auth", alias="DOCUMIND_AUTH_STATE_DIR")
    data_encryption_key: str | None = Field(default=None, alias="DOCUMIND_DATA_ENCRYPTION_KEY")
    public_registration_role: str = Field(default="VIEWER", alias="PUBLIC_REGISTRATION_ROLE")
    rate_limit_requests: int = Field(default=180, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")
    trust_proxy_headers: bool = Field(default=False, alias="TRUST_PROXY_HEADERS")
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    frontend_port: int = Field(default=5174, alias="FRONTEND_PORT")
    enable_chroma: bool = Field(default=False, alias="ENABLE_CHROMA")
    enable_neo4j: bool = Field(default=False, alias="ENABLE_NEO4J")
    enable_mongodb: bool = Field(default=False, alias="ENABLE_MONGODB")
    enable_redis: bool = Field(default=False, alias="ENABLE_REDIS")
    enable_minio: bool = Field(default=False, alias="ENABLE_MINIO")
    model_profile: str = Field(default="LITE", alias="MODEL_PROFILE")
    enable_heavy_models: bool = Field(default=False, alias="ENABLE_HEAVY_MODELS")
    enable_paddleocr: bool = Field(default=False, alias="ENABLE_PADDLEOCR")
    enable_paddleocr_structure: bool = Field(default=False, alias="ENABLE_PADDLEOCR_STRUCTURE")
    enable_paddleocr_vl: bool = Field(default=False, alias="ENABLE_PADDLEOCR_VL")
    enable_layoutlmv3: bool = Field(default=False, alias="ENABLE_LAYOUTLMV3")
    enable_donut: bool = Field(default=False, alias="ENABLE_DONUT")
    enable_yolo26: bool = Field(default=False, alias="ENABLE_YOLO26")
    enable_dinov2: bool = Field(default=False, alias="ENABLE_DINOV2")
    enable_bge_m3: bool = Field(default=False, alias="ENABLE_BGE_M3")
    enable_qdrant: bool = Field(default=False, alias="ENABLE_QDRANT")
    enable_qwen_local: bool = Field(default=False, alias="ENABLE_QWEN_LOCAL")
    qwen_allow_download: bool = Field(default=False, alias="QWEN_ALLOW_DOWNLOAD")
    bge_allow_download: bool = Field(default=False, alias="BGE_ALLOW_DOWNLOAD")
    enable_external_llm: bool = Field(default=False, alias="ENABLE_EXTERNAL_LLM")
    enable_demo_api: bool = Field(default=False, alias="ENABLE_DEMO_API")
    external_llm_base_url: str | None = Field(default=None, alias="EXTERNAL_LLM_BASE_URL")
    external_llm_api_key: str | None = Field(default=None, alias="EXTERNAL_LLM_API_KEY")
    external_llm_model: str | None = Field(default=None, alias="EXTERNAL_LLM_MODEL")
    ocr_low_confidence_threshold: float = Field(default=0.70, alias="OCR_LOW_CONFIDENCE_THRESHOLD")
    reading_order_low_confidence_threshold: float = Field(
        default=0.75,
        alias="READING_ORDER_LOW_CONFIDENCE_THRESHOLD",
    )
    language_min_valid_characters: int = Field(
        default=40,
        alias="LANGUAGE_MIN_VALID_CHARACTERS",
    )
    language_secondary_ratio: float = Field(
        default=0.20,
        alias="LANGUAGE_SECONDARY_RATIO",
    )
    language_secondary_min_characters: int = Field(
        default=20,
        alias="LANGUAGE_SECONDARY_MIN_CHARACTERS",
    )
    language_confidence_threshold: float = Field(
        default=0.80,
        alias="LANGUAGE_CONFIDENCE_THRESHOLD",
    )
    review_reauthentication_ttl_seconds: int = Field(
        default=5 * 60,
        alias="REVIEW_REAUTHENTICATION_TTL_SECONDS",
    )
    fraud_second_reviewer_required: bool = Field(
        default=False,
        alias="FRAUD_SECOND_REVIEWER_REQUIRED",
    )
    tamper_screening_review_threshold: float = Field(
        default=0.92,
        alias="TAMPER_SCREENING_REVIEW_THRESHOLD",
    )
    visual_duplicate_threshold: float = Field(
        default=0.92,
        alias="VISUAL_DUPLICATE_THRESHOLD",
    )
    dinov2_duplicate_threshold: float = Field(
        default=0.94,
        alias="DINOV2_DUPLICATE_THRESHOLD",
    )
    related_document_threshold: float = Field(
        default=0.62,
        alias="RELATED_DOCUMENT_THRESHOLD",
    )
    anomaly_min_samples: int = Field(default=20, alias="ANOMALY_MIN_SAMPLES")
    anomaly_review_percentile: float = Field(
        default=0.95,
        alias="ANOMALY_REVIEW_PERCENTILE",
    )
    xgboost_risk_checkpoint: str | None = Field(default=None, alias="XGBOOST_RISK_CHECKPOINT")
    xgboost_review_threshold: float = Field(default=0.5, alias="XGBOOST_REVIEW_THRESHOLD")
    contract_clause_threshold: float = Field(default=0.5, alias="CONTRACT_CLAUSE_THRESHOLD")
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"]
    )
    role_labels: list[str] = Field(
        default=[
            "ADMIN",
            "PROCESSOR",
            "REVIEWER",
            "SENIOR_REVIEWER",
            "COMPLIANCE_OFFICER",
            "AUDITOR",
            "VIEWER",
        ]
    )
    upload_dir: str = Field(default="var/uploads", alias="UPLOAD_DIR")
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")
    allowed_file_extensions: list[str] = Field(
        default=[".pdf", ".docx", ".txt", ".eml", ".md", ".csv", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    )
    allowed_mime_types: list[str] = Field(
        default=[
            "application/pdf",
            "image/png",
            "image/jpeg",
            "image/tiff",
            "text/plain",
            "text/csv",
            "text/markdown",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "message/rfc822",
        ]
    )
    malware_scanner_command: str | None = Field(default=None, alias="MALWARE_SCANNER_COMMAND")

    pdf_render_dpi: int = Field(default=200, alias="PDF_RENDER_DPI")
    vision_artifact_dir: str = Field(default="var/vision_artifacts", alias="VISION_ARTIFACT_DIR")
    model_smoke_state_dir: str = Field(
        default="var/vision_state/model_smoke",
        alias="MODEL_SMOKE_STATE_DIR",
    )
    tesseract_command: str = Field(default="tesseract", alias="TESSERACT_COMMAND")
    tesseract_languages: str = Field(default="eng", alias="TESSERACT_LANGUAGES")
    paddleocr_device: str = Field(default="cpu", alias="PADDLEOCR_DEVICE")
    paddleocr_vl_engine: str = Field(default="paddle", alias="PADDLEOCR_VL_ENGINE")
    paddleocr_vl_pipeline_version: str = Field(
        default="v1.6",
        alias="PADDLEOCR_VL_PIPELINE_VERSION",
    )
    layoutlmv3_checkpoint: str | None = Field(default=None, alias="LAYOUTLMV3_CHECKPOINT")
    layoutlmv3_classification_checkpoint: str | None = Field(
        default=None,
        alias="LAYOUTLMV3_CLASSIFICATION_CHECKPOINT",
    )
    donut_checkpoint: str | None = Field(default=None, alias="DONUT_CHECKPOINT")
    yolo26_checkpoint: str | None = Field(default=None, alias="YOLO26_CHECKPOINT")
    dinov2_checkpoint: str | None = Field(default=None, alias="DINOV2_CHECKPOINT")
    contract_checkpoint: str | None = Field(default=None, alias="CONTRACT_CHECKPOINT")
    tamper_checkpoint: str | None = Field(default=None, alias="TAMPER_CHECKPOINT")
    bge_m3_checkpoint: str = Field(default="BAAI/bge-m3", alias="BGE_M3_CHECKPOINT")
    bge_reranker_checkpoint: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        alias="BGE_RERANKER_CHECKPOINT",
    )
    qwen_checkpoint: str = Field(default="Qwen/Qwen3.5-9B", alias="QWEN_CHECKPOINT")

    @model_validator(mode="after")
    def validate_production_secrets(self):
        if self.app_env.lower() in {"production", "prod"}:
            if not self.auth_secret or len(self.auth_secret) < 32:
                raise ValueError("DOCUMIND_AUTH_SECRET must contain at least 32 characters in production")
            if not self.data_encryption_key:
                raise ValueError("DOCUMIND_DATA_ENCRYPTION_KEY is required in production")
            if self.enable_minio and (not self.minio_access_key or not self.minio_secret_key):
                raise ValueError("MINIO credentials are required when MinIO is enabled")
            if self.enable_neo4j and (not self.neo4j_user or not self.neo4j_password):
                raise ValueError("Neo4j credentials are required when Neo4j is enabled")
        return self

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key or self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
