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
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1) from None
