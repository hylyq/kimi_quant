"""Deposit USDC from Arbitrum to Hyperliquid L1.

⚠️ WARNING — USE AT YOUR OWN RISK ⚠️

This module calls ERC20 `transfer` to the Hyperliquid Bridge2 contract.
The bridge MAY NOT accept plain transfers — it might require a specific
`deposit()` function call. If that's the case, USDC sent via this method
could be IRREVERSIBLY LOST.

The SAFEST way to deposit is the official UI:
  https://app.hyperliquid.xyz/trade → Deposit

This module is provided as a reference. Verify the bridge contract
interface against Hyperliquid's official documentation before using:
  https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/bridge2

Minimum deposit: 5 USDC. Amounts below this are lost by the bridge.
"""

import logging
import time

from eth_account import Account
from eth_account.signers.local import LocalAccount
from web3 import Web3
from web3.types import TxReceipt

from kimi_quant.config import config

logger = logging.getLogger(__name__)

# Arbitrum One
ARBITRUM_CHAIN_ID = 42161
ARBITRUM_RPC = "https://arb1.arbitrum.io/rpc"

# Native USDC on Arbitrum (Circle-issued)
USDC_ADDRESS = Web3.to_checksum_address(
    "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
)

# Hyperliquid Bridge2 (deposit bridge on Arbitrum)
BRIDGE_ADDRESS = Web3.to_checksum_address(
    "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7"
)

# Minimum deposit enforced by the bridge
MIN_DEPOSIT_USDC = 5.0

# Standard ERC20 ABI — just what we need
USDC_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "dst", "type": "address"},
            {"name": "wad", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "src", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]


def _get_account() -> LocalAccount:
    """Load the Hyperliquid wallet from config."""
    if not config.hl_private_key:
        raise ValueError(
            "HYPERLIQUID_PRIVATE_KEY is required. Set it in .env."
        )
    return Account.from_key(config.hl_private_key)


def check_balance(address: str | None = None) -> tuple[float, float]:
    """Check USDC and ETH balances on Arbitrum.

    Returns:
        (usdc_balance, eth_balance) in human-readable units.
    """
    w3 = Web3(Web3.HTTPProvider(ARBITRUM_RPC))
    acct = _get_account()
    addr = address or acct.address

    usdc_contract = w3.eth.contract(address=USDC_ADDRESS, abi=USDC_ABI)
    decimals: int = usdc_contract.functions.decimals().call()
    raw_balance: int = usdc_contract.functions.balanceOf(addr).call()
    usdc_balance = raw_balance / (10 ** decimals)

    eth_balance = w3.eth.get_balance(addr) / 1e18

    return usdc_balance, eth_balance


def deposit_usdc(amount: float) -> TxReceipt:
    """Deposit USDC from Arbitrum to Hyperliquid L1.

    Transfers USDC to the Hyperliquid bridge contract on Arbitrum.
    The bridge credits the sender's Hyperliquid L1 account automatically.

    Args:
        amount: Amount of USDC to deposit (min 5.0).

    Returns:
        The transaction receipt.
    """
    if amount < MIN_DEPOSIT_USDC:
        raise ValueError(
            f"Minimum deposit is {MIN_DEPOSIT_USDC} USDC. "
            f"Amounts below this are lost by the bridge."
        )

    w3 = Web3(Web3.HTTPProvider(ARBITRUM_RPC))
    acct = _get_account()

    usdc_balance, eth_balance = check_balance(acct.address)

    if usdc_balance < amount:
        raise ValueError(
            f"Insufficient USDC balance: {usdc_balance:.2f} < {amount:.2f}"
        )

    if eth_balance < 0.0001:
        logger.warning(
            "ETH balance on Arbitrum is very low (%.6f ETH). "
            "You may not have enough for gas. Consider bridging a small "
            "amount of ETH to Arbitrum first.",
            eth_balance,
        )

    usdc_contract = w3.eth.contract(address=USDC_ADDRESS, abi=USDC_ABI)
    decimals: int = usdc_contract.functions.decimals().call()
    raw_amount = int(amount * (10 ** decimals))

    # Build the transfer transaction
    tx = usdc_contract.functions.transfer(
        BRIDGE_ADDRESS, raw_amount
    ).build_transaction({
        "from": acct.address,
        "chainId": ARBITRUM_CHAIN_ID,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "gas": 100_000,  # ERC20 transfer ~60k gas, add buffer
        "gasPrice": w3.eth.gas_price,
    })

    logger.info(
        "Depositing %.2f USDC to Hyperliquid from %s via Arbitrum...",
        amount, acct.address,
    )

    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    logger.info("Tx sent: https://arbiscan.io/tx/%s", tx_hash.hex())

    # Wait for confirmation
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    if receipt["status"] == 1:
        logger.info(
            "Deposit successful! %.2f USDC → Hyperliquid (%s). "
            "It may take 1-2 minutes to appear in your perp account.",
            amount, acct.address,
        )
        logger.info(
            "Verify at: https://app.hyperliquid.xyz/trade"
        )
    else:
        logger.error("Deposit transaction failed! Check Arbiscan for details.")

    return receipt


def cmd_deposit(amount: float, force: bool = False) -> None:
    """CLI handler for deposit command."""
    try:
        usdc_bal, eth_bal = check_balance()
        acct = _get_account()
        print(f"Arbitrum balances: {usdc_bal:.2f} USDC | {eth_bal:.6f} ETH")
        print(f"From: {acct.address}")
        print(f"To:   {BRIDGE_ADDRESS} (Hyperliquid Bridge2)")
        print()
        print("⚠️  This tool is experimental. The SAFEST way is:")
        print("    https://app.hyperliquid.xyz/trade → Deposit")
        print()

        if amount <= 0:
            print(f"Usage: uv run kimi-quant --deposit <amount>")
            print(f"  Minimum deposit: {MIN_DEPOSIT_USDC} USDC")
            return

        if not force:
            resp = input(
                f"Deposit {amount:.2f} USDC to Hyperliquid Bridge2? "
                f"Type YES to confirm: "
            )
            if resp.strip() != "YES":
                print("Cancelled.")
                return

        print(f"Depositing {amount:.2f} USDC...")
        receipt = deposit_usdc(amount)
        print(f"Tx hash: {receipt['transactionHash'].hex()}")
        print("Done. Check Hyperliquid in 1-2 minutes.")
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from None


# ─── Account Type Management ─────────────────────────────────────────────


ACCOUNT_TYPE_LABELS = {
    "default": "Default",
    "disabled": "Manual (separate spot & perp)",
    "unifiedAccount": "Unified Account",
    "portfolioMargin": "Portfolio Margin",
    "dexAbstraction": "DEX Abstraction",
}


def get_account_type(address: str | None = None) -> str:
    """Query the current account abstraction mode."""
    from hyperliquid.info import Info

    acct = _get_account()
    addr = address or acct.address
    base_url = (
        "https://api.hyperliquid-testnet.xyz"
        if config.hl_testnet
        else "https://api.hyperliquid.xyz"
    )
    info = Info(base_url=base_url, skip_ws=True)
    state = info.query_user_abstraction_state(addr)
    return state or "default"


def set_account_type(mode: str) -> dict:
    """Change account abstraction mode.

    Args:
        mode: "manual" | "unified" | "portfolio"
    """
    from hyperliquid.exchange import Exchange

    mode_map = {
        "manual": "disabled",
        "unified": "unifiedAccount",
        "portfolio": "portfolioMargin",
    }
    hl_mode = mode_map.get(mode, mode)

    acct = _get_account()
    base_url = (
        "https://api.hyperliquid-testnet.xyz"
        if config.hl_testnet
        else "https://api.hyperliquid.xyz"
    )
    ex = Exchange(wallet=acct, base_url=base_url)

    logger.info(
        "Setting account type for %s → %s (%s)...",
        acct.address, mode, hl_mode,
    )
    result = ex.user_set_abstraction(acct.address, hl_mode)
    logger.info("Result: %s", result)
    return result


def cmd_set_account_type(mode: str, force: bool = False) -> None:
    """CLI handler for account type changes."""
    try:
        current = get_account_type()
        label = ACCOUNT_TYPE_LABELS.get(current, current)
        print(f"Current account type: {label} ({current})")
        print()

        valid_modes = ["manual", "unified", "portfolio"]
        if mode not in valid_modes:
            print(f"Usage: uv run kimi-quant --set-account-type <mode>")
            print(f"  Valid modes: {', '.join(valid_modes)}")
            print(f"  Recommended for kimi_quant: manual")
            return

        target_hl = {
            "manual": "disabled",
            "unified": "unifiedAccount",
            "portfolio": "portfolioMargin",
        }[mode]
        if current == target_hl:
            print(f"Account is already in {mode} mode. Nothing to change.")
            return

        target_label = ACCOUNT_TYPE_LABELS.get(target_hl, target_hl)
        if not force:
            resp = input(
                f"Change from {label} → {target_label}? Type YES to confirm: "
            )
            if resp.strip() != "YES":
                print("Cancelled.")
                return

        result = set_account_type(mode)
        print(f"Done: {result}")

        # Verify
        new_current = get_account_type()
        new_label = ACCOUNT_TYPE_LABELS.get(new_current, new_current)
        print(f"New account type: {new_label}")

    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1) from None


# ─── Spot ↔ Perp Transfer ────────────────────────────────────────────────


def get_spot_balance(address: str | None = None) -> dict[str, float]:
    """Get Hyperliquid spot account balances.

    Returns:
        Dict mapping coin name to available balance.
    """
    import requests
    acct = _get_account()
    addr = address or acct.address
    resp = requests.post(
        "https://api.hyperliquid.xyz/info",
        json={"type": "spotClearinghouseState", "user": addr},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    balances: dict[str, float] = {}
    for b in data.get("balances", []):
        coin = b["coin"]
        total = float(b["total"])
        if total > 0:
            balances[coin] = total
    return balances


def get_perp_balance(address: str | None = None) -> float:
    """Get Hyperliquid perp account value."""
    import requests
    acct = _get_account()
    addr = address or acct.address
    resp = requests.post(
        "https://api.hyperliquid.xyz/info",
        json={"type": "clearinghouseState", "user": addr},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data.get("marginSummary", {}).get("accountValue", "0"))


def spot_to_perp(amount: float) -> dict:
    """Transfer USDC from spot to perp account.

    For standard accounts: uses usd_class_transfer (L1 action, instant).
    For unified accounts: falls back to spot_transfer to self, or
    instructs the user to use the web UI.
    """
    from hyperliquid.exchange import Exchange

    if amount <= 0:
        raise ValueError(f"Amount must be positive, got {amount}")

    spot_balances = get_spot_balance()
    spot_usdc = spot_balances.get("USDC", 0.0)
    if spot_usdc < amount:
        raise ValueError(
            f"Insufficient spot USDC: {spot_usdc:.2f} < {amount:.2f}"
        )

    acct = _get_account()
    base_url = (
        "https://api.hyperliquid-testnet.xyz"
        if config.hl_testnet
        else "https://api.hyperliquid.xyz"
    )
    ex = Exchange(wallet=acct, base_url=base_url)

    logger.info("Transferring %.2f USDC spot → perp...", amount)

    # Try usd_class_transfer (works for standard accounts)
    result = ex.usd_class_transfer(amount, to_perp=True)
    logger.info("Result: %s", result)

    if result.get("status") == "ok":
        return result

    # Unified account: usd_class_transfer is disabled.
    # spot_transfer moves between users, not between spot/perp.
    # The SDK has no unified-account transfer method yet.
    if "unified account" in str(result).lower():
        raise RuntimeError(
            "Your account is a Unified Account, which blocks programmatic "
            "spot→perp transfers.\n\n"
            "Please use the web UI instead:\n"
            "  https://app.hyperliquid.xyz/trade\n"
            f"  → Click your balance → Transfer → Move {amount:.2f} USDC "
            "from Spot to Perpetual\n"
            "  → Then restart: uv run kimi-quant"
        )

    raise RuntimeError(f"Transfer failed: {result}")


def cmd_spot_to_perp(amount: float, force: bool = False) -> None:
    """CLI handler for spot→perp transfer."""
    try:
        spot_balances = get_spot_balance()
        perp_balance = get_perp_balance()

        print(f"Hyperliquid balances for {_get_account().address}:")
        print(f"  Spot:  ", end="")
        if spot_balances:
            for coin, bal in sorted(spot_balances.items()):
                print(f"{bal:.2f} {coin}  ", end="")
            print()
        else:
            print("(empty)")
        print(f"  Perp:  {perp_balance:.2f} USDC")
        print()

        if amount <= 0:
            print(f"Usage: uv run kimi-quant --spot-to-perp <amount>")
            return

        spot_usdc = spot_balances.get("USDC", 0.0)
        if spot_usdc < amount:
            print(f"Error: insufficient spot USDC ({spot_usdc:.2f})")
            raise SystemExit(1)

        if not force:
            resp = input(
                f"Transfer {amount:.2f} USDC from spot to perp? [y/N]: "
            )
            if resp.strip().lower() not in ("y", "yes"):
                print("Cancelled.")
                return

        result = spot_to_perp(amount)
        print(f"Done: {result}")

        # Verify
        new_perp = get_perp_balance()
        print(f"Perp balance now: {new_perp:.2f} USDC")

    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from None


# ─── Account Type Management ─────────────────────────────────────────────


ACCOUNT_TYPE_LABELS = {
    "default": "Default",
    "disabled": "Manual (separate spot & perp)",
    "unifiedAccount": "Unified Account",
    "portfolioMargin": "Portfolio Margin",
    "dexAbstraction": "DEX Abstraction",
}


def get_account_type(address: str | None = None) -> str:
    """Query the current account abstraction mode."""
    from hyperliquid.info import Info

    acct = _get_account()
    addr = address or acct.address
    base_url = (
        "https://api.hyperliquid-testnet.xyz"
        if config.hl_testnet
        else "https://api.hyperliquid.xyz"
    )
    info = Info(base_url=base_url, skip_ws=True)
    state = info.query_user_abstraction_state(addr)
    return state or "default"


def set_account_type(mode: str) -> dict:
    """Change account abstraction mode.

    Args:
        mode: "manual" | "unified" | "portfolio"
    """
    from hyperliquid.exchange import Exchange

    mode_map = {
        "manual": "disabled",
        "unified": "unifiedAccount",
        "portfolio": "portfolioMargin",
    }
    hl_mode = mode_map.get(mode, mode)

    acct = _get_account()
    base_url = (
        "https://api.hyperliquid-testnet.xyz"
        if config.hl_testnet
        else "https://api.hyperliquid.xyz"
    )
    ex = Exchange(wallet=acct, base_url=base_url)

    logger.info(
        "Setting account type for %s → %s (%s)...",
        acct.address, mode, hl_mode,
    )
    result = ex.user_set_abstraction(acct.address, hl_mode)
    logger.info("Result: %s", result)
    return result


def cmd_set_account_type(mode: str, force: bool = False) -> None:
    """CLI handler for account type changes."""
    try:
        current = get_account_type()
        label = ACCOUNT_TYPE_LABELS.get(current, current)
        print(f"Current account type: {label} ({current})")
        print()

        valid_modes = ["manual", "unified", "portfolio"]
        if mode not in valid_modes:
            print(f"Usage: uv run kimi-quant --set-account-type <mode>")
            print(f"  Valid modes: {', '.join(valid_modes)}")
            print(f"  Recommended for kimi_quant: manual")
            return

        target_hl = {
            "manual": "disabled",
            "unified": "unifiedAccount",
            "portfolio": "portfolioMargin",
        }[mode]
        if current == target_hl:
            print(f"Account is already in {mode} mode. Nothing to change.")
            return

        target_label = ACCOUNT_TYPE_LABELS.get(target_hl, target_hl)
        if not force:
            resp = input(
                f"Change from {label} → {target_label}? Type YES to confirm: "
            )
            if resp.strip() != "YES":
                print("Cancelled.")
                return

        result = set_account_type(mode)
        print(f"Done: {result}")

        # Verify
        new_current = get_account_type()
        new_label = ACCOUNT_TYPE_LABELS.get(new_current, new_current)
        print(f"New account type: {new_label}")

    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1) from None
