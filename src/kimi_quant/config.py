"""Configuration management for Kimi Quant.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (src/kimi_quant/ → src/ → ../)
load_dotenv(Path(__file__).parent.parent.parent / ".env")


@dataclass
class Config:
    """Application configuration."""

    # --- Kimi / Moonshot API (Primary) ---
    moonshot_api_key: str = field(
        default_factory=lambda: os.getenv("MOONSHOT_API_KEY", "")
    )
    moonshot_base_url: str = field(
        default_factory=lambda: os.getenv(
            "MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"
        )
    )
    kimi_model: str = field(
        default_factory=lambda: os.getenv("KIMI_MODEL", "kimi-k3")
    )

    # --- DeepSeek API (Fallback) ---
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
        )
    )
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v3.1")
    )

    # --- LLM Parameters ---
    primary_llm: str = field(
        default_factory=lambda: os.getenv("PRIMARY_LLM", "kimi")
    )  # "kimi" or "deepseek" — which model to try first
    reasoning_effort: str = field(
        default_factory=lambda: os.getenv("REASONING_EFFORT", "max")
    )  # "max" | "high" | "medium" | "low" | "minimal" | "off"
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )

    # --- Hyperliquid ---
    hl_private_key: str = field(
        default_factory=lambda: os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    )
    hl_base_url: str = field(
        default_factory=lambda: os.getenv(
            "HYPERLIQUID_BASE_URL", "https://api.hyperliquid.xyz"
        )
    )
    hl_testnet: bool = field(
        default_factory=lambda: os.getenv("HYPERLIQUID_TESTNET", "true").lower()
        == "true"
    )

    # --- Trading Parameters ---
    trading_pair: str = field(
        default_factory=lambda: os.getenv("TRADING_PAIR", "BTC")
    )
    max_position_size: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE", "0.01"))
    )
    min_confidence: float = field(
        default_factory=lambda: float(os.getenv("MIN_CONFIDENCE", "0.7"))
    )
    max_leverage: int = field(
        default_factory=lambda: int(os.getenv("MAX_LEVERAGE", "3"))
    )

    # --- Strategy ---
    strategy_mode: str = field(
        default_factory=lambda: os.getenv("STRATEGY_MODE", "single")
    )  # "single" | "debate"
    judge_temperature: float = field(
        default_factory=lambda: float(os.getenv("JUDGE_TEMPERATURE", "0.05"))
    )
    trading_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("TRADING_INTERVAL", "600"))
    )
    min_interval: int = field(
        default_factory=lambda: int(os.getenv("MIN_INTERVAL", "300"))
    )  # hard lower bound for LLM-suggested intervals (default 5 min)
    max_interval: int = field(
        default_factory=lambda: int(os.getenv("MAX_INTERVAL", "10800"))
    )  # hard upper bound (default 3 hours)
    dry_run: bool = field(
        default_factory=lambda: os.getenv("DRY_RUN", "true").lower() == "true"
    )

    # --- Logging ---
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    def validate(self) -> None:
        """Validate required configuration."""
        if not self.moonshot_api_key:
            raise ValueError(
                "MOONSHOT_API_KEY is required. Set it in .env or environment."
            )
        if not self.dry_run and not self.hl_private_key:
            raise ValueError(
                "HYPERLIQUID_PRIVATE_KEY is required for live trading. "
                "Set it in .env or use DRY_RUN=true."
            )

    @property
    def display_model(self) -> str:
        """The model name to show in logs/banners, based on PRIMARY_LLM."""
        if self.primary_llm.lower() == "deepseek" and self.deepseek_api_key:
            return self.deepseek_model
        return self.kimi_model


# Singleton config instance
config = Config()
