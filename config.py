from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    google_api_key: str = Field(default="", description="Google API key for Gemini LLM")
    marker_api_key: str = Field(default="", description="Marker API key for OCR service")
    
    # LLM Settings
    llm_model: str = Field(default="gemini-2.5-flash", description="LLM model for main parsing")
    llm_temperature: float = Field(default=0.0, description="LLM temperature setting")
    
    # LLM Utility Settings (for table extraction, matching, unit detection)
    llm_utils_model: str = Field(default="gemini-2.5-flash", description="Fast model for utilities")
    llm_use_for_matching: bool = Field(default=True, description="Use LLM for item matching")
    llm_use_for_extraction: bool = Field(default=True, description="Use LLM for table extraction")
    llm_table_extraction_threshold: int = Field(default=80000, description="Document size threshold for table extraction (chars)")
    
    # OCR Settings
    default_ocr_service: Literal["marker", "docling", "vintern", "paddle"] = Field(
        default="marker", description="Default OCR service to use"
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
