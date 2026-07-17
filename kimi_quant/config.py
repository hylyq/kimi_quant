"""Configuration management for Kimi Quant.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present
load_dotenv(Path(__file__).parent.parent / ".env")


@dataclass
class Config:
    """Application configuration."""

    # --- Kimi / Moonshot API ---
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
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024"))
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
    trading_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("TRADING_INTERVAL", "300"))
    )
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


# Singleton config instance
config = Config()
