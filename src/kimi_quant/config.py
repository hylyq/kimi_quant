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
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
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
    judge_primary_llm: str = field(
        default_factory=lambda: os.getenv("JUDGE_PRIMARY_LLM", "")
    )  # "" = use PRIMARY_LLM; set to "kimi" or "deepseek" to override Judge model
    debate_rebuttal_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "DEBATE_REBUTTAL_ENABLED", "false"
        ).lower() == "true"
    )  # enables a rebuttal round where debaters counter each other before Judge rules
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

    # --- Cache ---
    cache_warmup_delay: float = field(
        default_factory=lambda: float(os.getenv("CACHE_WARMUP_DELAY", "2.0"))
    )  # Seconds to wait after cache-warmup agent completes (debate mode).
    # Gives DeepSeek disk cache time to flush before subsequent agents fire.
    # Set to 0 to disable.

    # --- Order Monitor (Real-time WebSocket + Flash LLM reporting) ---
    monitor_enabled: bool = field(
        default_factory=lambda: os.getenv("MONITOR_ENABLED", "true").lower()
        == "true"
    )
    monitor_flash_model: str = field(
        default_factory=lambda: os.getenv(
            "MONITOR_FLASH_MODEL", "deepseek-v4-flash"
        )
    )
    monitor_flash_base_url: str = field(
        default_factory=lambda: os.getenv(
            "MONITOR_FLASH_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        )
    )
    monitor_flash_api_key: str = field(
        default_factory=lambda: os.getenv(
            "MONITOR_FLASH_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")
        )
    )

    # --- Logging ---
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    def validate(self) -> None:
        """Validate required configuration and value ranges."""
        errors: list[str] = []

        # API keys
        if not self.moonshot_api_key and not self.deepseek_api_key:
            errors.append(
                "At least one LLM API key is required. "
                "Set MOONSHOT_API_KEY or DEEPSEEK_API_KEY in .env"
            )
        if not self.dry_run and not self.hl_private_key:
            errors.append(
                "HYPERLIQUID_PRIVATE_KEY is required for live trading. "
                "Set it in .env or use DRY_RUN=true."
            )

        # LLM settings
        if self.primary_llm not in ("kimi", "deepseek"):
            errors.append(
                f"PRIMARY_LLM must be 'kimi' or 'deepseek', got '{self.primary_llm}'"
            )
        if self.judge_primary_llm and self.judge_primary_llm not in ("kimi", "deepseek"):
            errors.append(
                f"JUDGE_PRIMARY_LLM must be 'kimi' or 'deepseek', got '{self.judge_primary_llm}'"
            )
        if self.llm_temperature < 0:
            errors.append(f"LLM_TEMPERATURE must be >= 0, got {self.llm_temperature}")
        if self.llm_max_tokens <= 0:
            errors.append(f"LLM_MAX_TOKENS must be > 0, got {self.llm_max_tokens}")
        if self.judge_temperature < 0:
            errors.append(f"JUDGE_TEMPERATURE must be >= 0, got {self.judge_temperature}")

        # Strategy
        if self.strategy_mode not in ("single", "debate"):
            errors.append(
                f"STRATEGY_MODE must be 'single' or 'debate', got '{self.strategy_mode}'"
            )

        # Trading parameters
        if not self.trading_pair:
            errors.append("TRADING_PAIR must not be empty")
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            errors.append(
                f"MIN_CONFIDENCE must be in [0.0, 1.0], got {self.min_confidence}"
            )
        if self.max_leverage <= 0:
            errors.append(f"MAX_LEVERAGE must be > 0, got {self.max_leverage}")
        if self.max_position_size <= 0:
            errors.append(f"MAX_POSITION_SIZE must be > 0, got {self.max_position_size}")

        # Interval
        if self.trading_interval_seconds <= 0:
            errors.append(
                f"TRADING_INTERVAL must be > 0, got {self.trading_interval_seconds}"
            )
        if self.min_interval <= 0:
            errors.append(f"MIN_INTERVAL must be > 0, got {self.min_interval}")
        if self.max_interval < self.min_interval:
            errors.append(
                f"MAX_INTERVAL ({self.max_interval}) must be >= "
                f"MIN_INTERVAL ({self.min_interval})"
            )
        if self.cache_warmup_delay < 0:
            errors.append(
                f"CACHE_WARMUP_DELAY must be >= 0, got {self.cache_warmup_delay}"
            )

        if errors:
            raise ValueError(
                "Configuration errors:\n  - " + "\n  - ".join(errors)
            )

    @property
    def display_model(self) -> str:
        """The model name to show in logs/banners, based on PRIMARY_LLM."""
        if self.primary_llm.lower() == "deepseek" and self.deepseek_api_key:
            return self.deepseek_model
        if self.moonshot_api_key:
            return self.kimi_model
        return self.deepseek_model  # fallback: DeepSeek only


# Singleton config instance
config = Config()
