from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache

from llm_settings import DEFAULT_LLM_MODEL

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    marker_api_key: str = Field(default="", description="Marker API key for OCR service")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key for LLMs")
    
    # LLM Settings
    llm_model: str = Field(default=DEFAULT_LLM_MODEL, description="LLM model for all OpenRouter-backed tasks")
    llm_temperature: float = Field(default=0.0, description="LLM temperature setting")
    
    # Deprecated compatibility field: the codebase now uses llm_model for all tasks.
    # Keep the setting so older env files do not break.
    llm_utils_model: str = Field(default=DEFAULT_LLM_MODEL, description="Deprecated: use llm_model instead")
    llm_use_for_matching: bool = Field(default=True, description="Use LLM for item matching")
    
    # OCR Settings
    default_ocr_service: Literal["hybrid", "docling", "marker"] = Field(
        default="hybrid", description="Default OCR service to use"
    )
    ocr_max_polls: int = Field(default=175, description="Maximum polling attempts for OCR")
    ocr_poll_interval: int = Field(default=2, description="Polling interval in seconds")
    
    # Scraping Settings
    scraper_timeout: int = Field(default=30000, description="Playwright timeout in ms")
    scraper_wait_timeout: int = Field(default=60000, description="Wait for function timeout in ms")
    scraper_headless: bool = Field(default=True, description="Run browser in headless mode")
    
    # Retry Settings
    retry_max_attempts: int = Field(default=3, description="Maximum retry attempts")
    retry_min_wait: int = Field(default=2, description="Minimum wait between retries in seconds")
    retry_max_wait: int = Field(default=10, description="Maximum wait between retries in seconds")
    
    # Output Settings
    reports_output_dir: str = Field(default="data/reports", description="Directory for OCR output")
    parsed_output_dir: str = Field(default="data/parsed", description="Directory for parsed JSON")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Logging format"
    )
    
    # URLs
    vietstock_base_url: str = Field(
        default="https://finance.vietstock.vn",
        description="Vietstock base URL"
    )
    
    # UI Settings
    ui_max_reports: int = Field(default=1, description="Maximum reports to process")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

settings = get_settings()

# OCR Engine Options for UI
OCR_ENGINE_OPTIONS = [
    ("Hybrid (Docling + Surya)", "hybrid"),
    ("Docling", "docling"),
    ("Marker", "marker"),
]

# Financial Report Constants
class ReportConstants:
    """Constants for financial report processing."""
    
    # Period types
    PERIOD_QUARTER = "Quý"
    PERIOD_HALF_YEAR = "6 tháng"
    PERIOD_FULL_YEAR = "Cả năm"
    PERIOD_LATEST = "Mới nhất"
    
    # Consolidation status
    CONSOLIDATED = "Hợp nhất"
    PARENT_COMPANY = "Công ty mẹ"
    
    # Report section codes
    BS_TOTAL_ASSETS_CODE = "270"
    BS_TOTAL_RESOURCES_CODE = "440"
    PL_NET_PROFIT_CODE = "60"
    PL_PROFIT_BEFORE_TAX_CODE = "50"
    PL_TAX_CURRENT_CODE = "51"
    PL_TAX_DEFERRED_CODE = "52"
    CF_CASH_END_CODE = "70"
    CF_CASH_BEGIN_CODE = "60"
    CF_NET_CASH_FLOW_CODE = "50"
    CF_EXCHANGE_DIFF_CODE = "61"
    
    # Validation tolerance
    VALIDATION_TOLERANCE = 1.0

# Quarter to month mapping
QUARTER_END_MONTHS = {
    1: 3,   # Q1 ends in March
    2: 6,   # Q2 ends in June
    3: 9,   # Q3 ends in September
    4: 12   # Q4 ends in December
}
