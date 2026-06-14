"""
Day 073: MEV Detection Script

Detects three major types of MEV (Maximal Extractable Value) in Ethereum blocks:
1. Sandwich attacks — front-run + victim + back-run pattern
2. Arbitrage — circular swap paths exploiting price differences
3. Liquidations — seizing undercollateralized positions at a discount

We simulate realistic block data with embedded MEV patterns to demonstrate
detection without requiring a live Ethereum node.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math
import statistics


# =============================================================================
# Data Models
# =============================================================================

class MEVType(Enum):
    """Categories of MEV extraction strategies."""
    SANDWICH = "sandwich"
    ARBITRAGE = "arbitrage"
    LIQUIDATION = "liquidation"


@dataclass
class Token:
    """Represents an ERC-20 token."""
    symbol: str
    address: str  # Simulated hex address
    decimals: int = 18


@dataclass
class DEXSwap:
    """
    A decoded DEX swap event extracted from a transaction.

    In production, you'd decode this from event logs (Swap events) or
    trace the transaction's internal calls. Here we represent it directly.
    """
    pool_address: str          # The liquidity pool contract
    token_in: Token            # Token being sold
    token_out: Token           # Token being bought
    amount_in: float           # Amount of token_in sold
    amount_out: float          # Amount of token_out received
    effective_price: float     # amount_out / amount_in — price of token_in in terms of token_out

    @property
    def inverse_price(self) -> float:
        """Price of token_out in terms of token_in."""
        return self.amount_in / self.amount_out if self.amount_out > 0 else float('inf')


@dataclass
class LiquidationCall:
    """
    A decoded liquidation event from a lending protocol.

    When a borrower's health factor drops below 1.0, liquidators can repay
    part of the debt and receive collateral at a discount (the liquidation bonus).
    """
    protocol: str              # e.g., "Aave", "Compound"
    borrower: str              # Address being liquidated
    debt_token: Token          # Token the liquidator repays
    collateral_token: Token    # Token the liquidator receives
    debt_repaid: float         # Amount of debt repaid
    collateral_seized: float   # Amount of collateral received
    liquidation_bonus_pct: float  # Bonus percentage (e.g., 5.0 = 5%)


@dataclass
class Transaction:
    """
    Represents an Ethereum transaction with decoded DEX/lending interactions.

    In production, you'd decode the raw transaction input data using ABI
    decoders and parse event logs. Here we pre-populate the decoded fields.
    """
    tx_hash: str
    sender: str                # msg.sender (the EOA or contract initiating)
    receiver: str              # The contract being called
    value_eth: float           # ETH value sent with the transaction
    gas_price_gwei: float      # Gas price in Gwei (or priority fee post-EIP-1559)
    gas_used: int              # Gas consumed
    block_position: int        # Index within the block (0 = first tx)

    # Decoded interactions — a transaction can contain multiple swaps
    # (e.g., multi-hop routes or arbitrage paths)
    swaps: list[DEXSwap] = field(default_factory=list)
    liquidation: Optional[LiquidationCall] = None

    @property
    def gas_cost_eth(self) -> float:
        """Total gas cost in ETH."""
        return self.gas_price_gwei * self.gas_used * 1e-9

    @property
    def gas_cost_usd(self) -> float:
        """Approximate gas cost in USD (assuming ETH = $3000)."""
        return self.gas_cost_eth * 3000


@dataclass
class Block:
    """A simplified Ethereum block containing ordered transactions."""
    number: int
    timestamp: int
    base_fee_gwei: float       # EIP-1559 base fee
    transactions: list[Transaction] = field(default_factory=list)

    @property
    def median_gas_price(self) -> float:
        """Median gas price across all transactions in the block."""
        prices = [tx.gas_price_gwei for tx in self.transactions]
        return statistics.median(prices) if prices else 0.0


# =============================================================================
# MEV Detection Results
# =============================================================================

@dataclass
class SandwichDetection:
    """A detected sandwich attack."""
    block_number: int
    attacker: str
    victim: str
    pool: str
    token_bought: str          # Token the attacker front-runs on
    frontrun_tx: str           # Attacker's buy tx hash
    victim_tx: str             # Victim's tx hash
    backrun_tx: str            # Attacker's sell tx hash
    attacker_buy_amount: float
    attacker_sell_amount: float
    profit_token_amount: float # Gross profit in the token
    gas_cost_eth: float        # Total gas for front + back run
    estimated_profit_usd: float

    def __str__(self) -> str:
        return (
            f"  SANDWICH ATTACK in block {self.block_number}\n"
            f"    Attacker: {self.attacker}\n"
            f"    Victim:   {self.victim}\n"
            f"    Pool:     {self.pool}\n"
            f"    Token:    {self.token_bought}\n"
            f"    Front-run: bought {self.attacker_buy_amount:.4f} ({self.frontrun_tx})\n"
            f"    Victim:    swapped ({self.victim_tx})\n"
            f"    Back-run:  sold {self.attacker_sell_amount:.4f} ({self.backrun_tx})\n"
            f"    Gross profit: {self.profit_token_amount:.4f} tokens\n"
            f"    Gas cost:     {self.gas_cost_eth:.6f} ETH\n"
            f"    Est. profit:  ${self.estimated_profit_usd:.2f}"
        )


@dataclass
class ArbitrageDetection:
    """A detected arbitrage opportunity that was executed."""
    block_number: int
    searcher: str
    path: list[str]            # Token path: [ETH, USDC, ETH] = circular
    venues: list[str]          # DEX pools used
    input_amount: float
    output_amount: float
    profit_amount: float       # output - input
    profit_token: str
    gas_cost_eth: float
    estimated_profit_usd: float
    tx_hash: str

    def __str__(self) -> str:
        path_str = " -> ".join(self.path)
        return (
            f"  ARBITRAGE in block {self.block_number}\n"
            f"    Searcher: {self.searcher}\n"
            f"    Path:     {path_str}\n"
            f"    Venues:   {', '.join(self.venues)}\n"
            f"    In:       {self.input_amount:.4f} {self.profit_token}\n"
            f"    Out:      {self.output_amount:.4f} {self.profit_token}\n"
            f"    Profit:   {self.profit_amount:.4f} {self.profit_token}\n"
            f"    Gas cost: {self.gas_cost_eth:.6f} ETH\n"
            f"    Est. profit: ${self.estimated_profit_usd:.2f}\n"
            f"    Tx: {self.tx_hash}"
        )


@dataclass
class LiquidationDetection:
    """A detected liquidation event."""
    block_number: int
    liquidator: str
    borrower: str
    protocol: str
    debt_token: str
    collateral_token: str
    debt_repaid: float
    collateral_seized: float
    bonus_pct: float
    bonus_value_usd: float
    gas_cost_eth: float
    tx_hash: str

    def __str__(self) -> str:
        return (
            f"  LIQUIDATION in block {self.block_number}\n"
            f"    Liquidator: {self.liquidator}\n"
            f"    Borrower:   {self.borrower}\n"
            f"    Protocol:   {self.protocol}\n"
            f"    Repaid:     {self.debt_repaid:.2f} {self.debt_token}\n"
            f"    Seized:     {self.collateral_seized:.4f} {self.collateral_token}\n"
            f"    Bonus:      {self.bonus_pct:.1f}% (${self.bonus_value_usd:.2f})\n"
            f"    Gas cost:   {self.gas_cost_eth:.6f} ETH\n"
            f"    Tx: {self.tx_hash}"
        )


# =============================================================================
# MEV Detection Engine
# =============================================================================

class MEVDetector:
    """
    Analyzes blocks to detect MEV extraction patterns.

    The detector works by pattern-matching on decoded transaction data.
    In production, you'd feed it data from an archive node or indexer
    (like Dune, Flipside, or a custom EVM tracer). Here we work with
    pre-decoded Transaction objects.

    Detection approach for each MEV type:

    1. Sandwiches: Group swaps by pool, scan for A-B-A sender patterns
       where A's first swap direction opposes A's second swap direction.

    2. Arbitrage: Look for transactions with 2+ swaps forming a cycle
       (same start and end token) with net positive output.

    3. Liquidations: Directly check for decoded liquidation calls and
       compute the bonus value.
    """

    def __init__(self, eth_price_usd: float = 3000.0):
        self.eth_price_usd = eth_price_usd
        self.sandwiches: list[SandwichDetection] = []
        self.arbitrages: list[ArbitrageDetection] = []
        self.liquidations: list[LiquidationDetection] = []

    def analyze_block(self, block: Block) -> None:
        """
        Run all detectors on a single block.

        Order matters here: we detect sandwiches first because they span
        multiple transactions and need positional analysis. Arbitrage and
        liquidation detection operate on individual transactions.
        """
        self._detect_sandwiches(block)
        self._detect_arbitrage(block)
        self._detect_liquidations(block)

    def _detect_sandwiches(self, block: Block) -> None:
        """
        Detect sandwich attacks by scanning for the characteristic three-tx pattern.

        Algorithm:
        1. Collect all transactions that contain DEX swaps
        2. Group them by pool address (sandwiches target a specific pool)
        3. Within each pool group, sorted by block position, look for:
           - Tx_i from address A buying token X
           - Tx_j (i < j) from address B (victim) also buying token X
           - Tx_k (j < k) from address A selling token X

        Key insight: The attacker's front-run and back-run MUST be from the same
        address (or the same contract), and they must bracket the victim's trade.
        The attacker buys before the victim and sells after — the victim's trade
        pushes the price up, making the attacker's sell profitable.

        We also check that the attacker's sell amount exceeds their buy amount
        (they actually profited), because not every bracket pattern is a sandwich.
        """
        # Step 1: Collect swap transactions, indexed by pool
        # pool_address -> [(block_position, sender, swap, tx_hash)]
        pool_swaps: dict[str, list[tuple[int, str, DEXSwap, str]]] = {}

        for tx in block.transactions:
            for swap in tx.swaps:
                pool_key = swap.pool_address
                if pool_key not in pool_swaps:
                    pool_swaps[pool_key] = []
                pool_swaps[pool_key].append((tx.block_position, tx.sender, swap, tx.tx_hash))

        # Step 2: For each pool, look for sandwich patterns
        for pool_addr, swaps in pool_swaps.items():
            # Sort by position in block (should already be sorted, but be safe)
            swaps.sort(key=lambda x: x[0])

            # Step 3: Try all possible (front, victim, back) triples
            # This is O(n^3) but n is small per pool per block (usually < 20)
            for i in range(len(swaps)):
                for j in range(i + 1, len(swaps)):
                    for k in range(j + 1, len(swaps)):
                        pos_f, sender_f, swap_f, hash_f = swaps[i]
                        pos_v, sender_v, swap_v, hash_v = swaps[j]
                        pos_b, sender_b, swap_b, hash_b = swaps[k]

                        # Check pattern: same attacker, different victim
                        if sender_f != sender_b:
                            continue
                        if sender_f == sender_v:
                            continue  # Victim must be different

                        # Check direction: attacker buys X first, victim buys X,
                        # attacker sells X last
                        # "Buys X" = token_out is X; "Sells X" = token_in is X
                        attacker_bought_token = swap_f.token_out.symbol
                        victim_bought_token = swap_v.token_out.symbol
                        attacker_sold_token = swap_b.token_in.symbol

                        # Attacker front-runs buying the same token victim wants,
                        # then back-runs by selling that token
                        if (attacker_bought_token == victim_bought_token == attacker_sold_token):
                            # Verify profit: attacker sells more than they bought
                            profit = swap_b.amount_in - swap_f.amount_out
                            if profit <= 0:
                                continue  # Not profitable, not a sandwich

                            # Calculate gas cost for attacker (front + back run)
                            attacker_txs = [
                                tx for tx in block.transactions
                                if tx.tx_hash in (hash_f, hash_b)
                            ]
                            gas_cost = sum(tx.gas_cost_eth for tx in attacker_txs)

                            # Estimate profit in USD
                            # Use the effective price from the swap to convert
                            token_price_usd = swap_b.effective_price  # price in terms of token_in
                            profit_usd = profit * token_price_usd - gas_cost * self.eth_price_usd

                            self.sandwiches.append(SandwichDetection(
                                block_number=block.number,
                                attacker=sender_f,
                                victim=sender_v,
                                pool=pool_addr,
                                token_bought=attacker_bought_token,
                                frontrun_tx=hash_f,
                                victim_tx=hash_v,
                                backrun_tx=hash_b,
                                attacker_buy_amount=swap_f.amount_out,
                                attacker_sell_amount=swap_b.amount_in,
                                profit_token_amount=profit,
                                gas_cost_eth=gas_cost,
                                estimated_profit_usd=profit_usd,
                            ))

    def _detect_arbitrage(self, block: Block) -> None:
        """
        Detect arbitrage by finding circular swap paths within a single transaction.

        Algorithm:
        1. For each transaction with 2+ swaps, trace the token flow
        2. Check if the path is circular (starts and ends with same token)
        3. Verify net positive outcome (output > input)

        Why single-transaction? Atomic arbitrage (all swaps in one tx) is the
        dominant form because it's risk-free — if any leg fails, the entire
        transaction reverts. Multi-transaction arbitrage exists but is rare
        because it carries inventory risk.

        Path construction: We follow the chain of swaps where each swap's
        token_out matches the next swap's token_in. If we end up with the
        same token we started with and have more of it, it's arbitrage.
        """
        for tx in block.transactions:
            if len(tx.swaps) < 2:
                continue

            # Build the path by following token flow
            swaps = tx.swaps
            path_tokens = [swaps[0].token_in.symbol]
            venues = []

            # Check if swaps form a chain: swap[i].token_out == swap[i+1].token_in
            is_chain = True
            for idx in range(len(swaps)):
                path_tokens.append(swaps[idx].token_out.symbol)
                venues.append(swaps[idx].pool_address)
                if idx < len(swaps) - 1:
                    if swaps[idx].token_out.symbol != swaps[idx + 1].token_in.symbol:
                        is_chain = False
                        break

            if not is_chain:
                continue

            # Check circularity: start token == end token
            start_token = path_tokens[0]
            end_token = path_tokens[-1]
            if start_token != end_token:
                continue

            # Calculate profit: final output - initial input
            input_amount = swaps[0].amount_in
            output_amount = swaps[-1].amount_out
            profit = output_amount - input_amount

            if profit <= 0:
                continue  # Not profitable

            # Estimate USD value
            # If the token is ETH/WETH, use ETH price directly
            if start_token in ("ETH", "WETH"):
                profit_usd = profit * self.eth_price_usd
            else:
                # Rough estimate using the first swap's price
                profit_usd = profit * swaps[0].effective_price

            profit_usd -= tx.gas_cost_eth * self.eth_price_usd

            self.arbitrages.append(ArbitrageDetection(
                block_number=block.number,
                searcher=tx.sender,
                path=path_tokens,
                venues=venues,
                input_amount=input_amount,
                output_amount=output_amount,
                profit_amount=profit,
                profit_token=start_token,
                gas_cost_eth=tx.gas_cost_eth,
                estimated_profit_usd=profit_usd,
                tx_hash=tx.tx_hash,
            ))

    def _detect_liquidations(self, block: Block) -> None:
        """
        Detect liquidation events.

        This is simpler than sandwich/arb detection because liquidations
        are explicit contract calls. We just check for decoded liquidation
        data on each transaction.

        The profit for the liquidator is the liquidation bonus — typically
        5-10% of the collateral value. For example, if a borrower has 10 ETH
        collateral and 25,000 USDC debt with a 5% bonus, the liquidator
        repays up to 50% of the debt (12,500 USDC) and receives
        12,500 * 1.05 / 3000 = 4.375 ETH (worth $13,125, a $625 profit).
        """
        for tx in block.transactions:
            if tx.liquidation is None:
                continue

            liq = tx.liquidation

            # Calculate the bonus value in USD
            # The bonus is the extra collateral received beyond the debt value
            if liq.collateral_token.symbol in ("ETH", "WETH"):
                collateral_value_usd = liq.collateral_seized * self.eth_price_usd
            else:
                # For simplicity, estimate using a rough price
                collateral_value_usd = liq.collateral_seized * 1.0  # stablecoins

            bonus_value_usd = collateral_value_usd * (liq.liquidation_bonus_pct / 100.0)

            self.liquidations.append(LiquidationDetection(
                block_number=block.number,
                liquidator=tx.sender,
                borrower=liq.borrower,
                protocol=liq.protocol,
                debt_token=liq.debt_token.symbol,
                collateral_token=liq.collateral_token.symbol,
                debt_repaid=liq.debt_repaid,
                collateral_seized=liq.collateral_seized,
                bonus_pct=liq.liquidation_bonus_pct,
                bonus_value_usd=bonus_value_usd,
                gas_cost_eth=tx.gas_cost_eth,
                tx_hash=tx.tx_hash,
            ))

    def generate_report(self) -> str:
        """Generate a comprehensive MEV analysis report."""
        lines = []
        lines.append("=" * 70)
        lines.append("MEV DETECTION REPORT")
        lines.append("=" * 70)

        # Summary
        total_mev_usd = (
            sum(s.estimated_profit_usd for s in self.sandwiches) +
            sum(a.estimated_profit_usd for a in self.arbitrages) +
            sum(l.bonus_value_usd for l in self.liquidations)
        )
        lines.append(f"\nTotal MEV events detected: {len(self.sandwiches) + len(self.arbitrages) + len(self.liquidations)}")
        lines.append(f"Estimated total MEV value: ${total_mev_usd:,.2f}")

        # Sandwiches
        lines.append(f"\n--- SANDWICH ATTACKS ({len(self.sandwiches)} detected) ---")
        if self.sandwiches:
            total_sandwich_profit = sum(s.estimated_profit_usd for s in self.sandwiches)
            lines.append(f"Total estimated profit: ${total_sandwich_profit:,.2f}")
            for s in self.sandwiches:
                lines.append(str(s))
        else:
            lines.append("  None detected.")

        # Arbitrage
        lines.append(f"\n--- ARBITRAGE ({len(self.arbitrages)} detected) ---")
        if self.arbitrages:
            total_arb_profit = sum(a.estimated_profit_usd for a in self.arbitrages)
            lines.append(f"Total estimated profit: ${total_arb_profit:,.2f}")
            for a in self.arbitrages:
                lines.append(str(a))
        else:
            lines.append("  None detected.")

        # Liquidations
        lines.append(f"\n--- LIQUIDATIONS ({len(self.liquidations)} detected) ---")
        if self.liquidations:
            total_liq_bonus = sum(l.bonus_value_usd for l in self.liquidations)
            lines.append(f"Total liquidation bonus value: ${total_liq_bonus:,.2f}")
            for l in self.liquidations:
                lines.append(str(l))
        else:
            lines.append("  None detected.")

        # Gas analysis
        lines.append("\n--- GAS ANALYSIS ---")
        all_gas = (
            [s.gas_cost_eth for s in self.sandwiches] +
            [a.gas_cost_eth for a in self.arbitrages] +
            [l.gas_cost_eth for l in self.liquidations]
        )
        if all_gas:
            lines.append(f"Total gas spent on MEV: {sum(all_gas):.6f} ETH (${sum(all_gas) * self.eth_price_usd:,.2f})")
            lines.append(f"Average gas per MEV tx: {statistics.mean(all_gas):.6f} ETH")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# =============================================================================
# Simulated Block Data
# =============================================================================

def create_sample_tokens() -> dict[str, Token]:
    """Create a set of common tokens for simulation."""
    return {
        "WETH": Token("WETH", "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"),
        "USDC": Token("USDC", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6),
        "DAI":  Token("DAI",  "0x6B175474E89094C44Da98b954EedeAC495271d0F"),
        "LINK": Token("LINK", "0x514910771AF9Ca656af840dff83E8264EcF986CA"),
        "UNI":  Token("UNI",  "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"),
    }


def create_sample_blocks() -> list[Block]:
    """
    Create simulated blocks with embedded MEV patterns.

    This is where we craft realistic scenarios that our detector should find.
    Each block demonstrates a different MEV pattern.
    """
    tokens = create_sample_tokens()

    blocks = []

    # =========================================================================
    # Block 1: Contains a SANDWICH ATTACK on WETH/USDC pool
    # =========================================================================
    # Scenario: User wants to swap 10,000 USDC for WETH on Uniswap.
    # Attacker front-runs by buying WETH, victim trades at worse price,
    # attacker back-runs by selling WETH at higher price.
    block1 = Block(number=18_500_001, timestamp=1700000000, base_fee_gwei=30.0)

    # Normal tx before the sandwich (noise)
    block1.transactions.append(Transaction(
        tx_hash="0xaaaa1111", sender="0xUser1", receiver="0xUniV3Router",
        value_eth=0.0, gas_price_gwei=31.0, gas_used=150_000, block_position=0,
        swaps=[DEXSwap(
            pool_address="0xPool_WETH_USDC",
            token_in=tokens["USDC"], token_out=tokens["WETH"],
            amount_in=500.0, amount_out=0.167,
            effective_price=0.000334,  # USDC -> WETH price
        )],
    ))

    # FRONT-RUN: Attacker buys WETH with USDC (high gas to get positioned first)
    block1.transactions.append(Transaction(
        tx_hash="0xATTACK_FRONT", sender="0xMEVBot_Alpha", receiver="0xUniV3Router",
        value_eth=0.0, gas_price_gwei=80.0, gas_used=180_000, block_position=1,
        swaps=[DEXSwap(
            pool_address="0xPool_WETH_USDC",
            token_in=tokens["USDC"], token_out=tokens["WETH"],
            amount_in=50_000.0, amount_out=16.70,
            effective_price=0.000334,
        )],
    ))

    # VICTIM: User swaps USDC for WETH (at now-worse price due to front-run)
    block1.transactions.append(Transaction(
        tx_hash="0xVICTIM_001", sender="0xInnocentUser42", receiver="0xUniV3Router",
        value_eth=0.0, gas_price_gwei=35.0, gas_used=160_000, block_position=2,
        swaps=[DEXSwap(
            pool_address="0xPool_WETH_USDC",
            token_in=tokens["USDC"], token_out=tokens["WETH"],
            amount_in=10_000.0, amount_out=3.28,
            effective_price=0.000328,  # Worse price than market!
        )],
    ))

    # BACK-RUN: Attacker sells WETH for USDC at the inflated price
    block1.transactions.append(Transaction(
        tx_hash="0xATTACK_BACK", sender="0xMEVBot_Alpha", receiver="0xUniV3Router",
        value_eth=0.0, gas_price_gwei=78.0, gas_used=175_000, block_position=3,
        swaps=[DEXSwap(
            pool_address="0xPool_WETH_USDC",
            token_in=tokens["WETH"], token_out=tokens["USDC"],
            amount_in=16.85,  # Selling slightly more WETH value than bought
            amount_out=50_980.0,
            effective_price=3025.52,  # WETH -> USDC price (higher than buy)
        )],
    ))

    # Normal tx after sandwich (noise)
    block1.transactions.append(Transaction(
        tx_hash="0xaaaa2222", sender="0xUser2", receiver="0xSushiRouter",
        value_eth=1.0, gas_price_gwei=32.0, gas_used=120_000, block_position=4,
    ))

    blocks.append(block1)

    # =========================================================================
    # Block 2: Contains ARBITRAGE across Uniswap and Sushiswap
    # =========================================================================
    # Scenario: LINK is $15.10 on Uniswap but $15.30 on Sushiswap.
    # Searcher buys LINK on Uni, sells on Sushi, profits the spread.
    block2 = Block(number=18_500_002, timestamp=1700000012, base_fee_gwei=28.0)

    # Normal txs
    block2.transactions.append(Transaction(
        tx_hash="0xbbbb1111", sender="0xUser3", receiver="0xUniV3Router",
        value_eth=0.5, gas_price_gwei=30.0, gas_used=100_000, block_position=0,
    ))

    # ARBITRAGE: Single tx with multi-hop swap forming a cycle
    # WETH -> LINK (buy cheap on Uni) -> WETH (sell expensive on Sushi)
    block2.transactions.append(Transaction(
        tx_hash="0xARB_001", sender="0xArbSearcher99", receiver="0xArbContract",
        value_eth=0.0, gas_price_gwei=55.0, gas_used=250_000, block_position=1,
        swaps=[
            DEXSwap(
                pool_address="0xUni_WETH_LINK",
                token_in=tokens["WETH"], token_out=tokens["LINK"],
                amount_in=10.0, amount_out=1986.75,
                effective_price=198.675,  # 1 WETH = 198.675 LINK on Uni
            ),
            DEXSwap(
                pool_address="0xSushi_LINK_WETH",
                token_in=tokens["LINK"], token_out=tokens["WETH"],
                amount_in=1986.75, amount_out=10.08,
                effective_price=0.005074,  # 1 LINK = 0.005074 WETH on Sushi
            ),
        ],
    ))

    # More normal txs
    block2.transactions.append(Transaction(
        tx_hash="0xbbbb2222", sender="0xUser4", receiver="0xUniV3Router",
        value_eth=0.0, gas_price_gwei=29.0, gas_used=140_000, block_position=2,
        swaps=[DEXSwap(
            pool_address="0xUni_WETH_USDC",
            token_in=tokens["WETH"], token_out=tokens["USDC"],
            amount_in=2.0, amount_out=6010.0,
            effective_price=3005.0,
        )],
    ))

    blocks.append(block2)

    # =========================================================================
    # Block 3: Contains a LIQUIDATION on Aave
    # =========================================================================
    # Scenario: Borrower deposited WETH, borrowed DAI. ETH price dropped,
    # health factor fell below 1.0. Liquidator repays DAI, seizes WETH.
    block3 = Block(number=18_500_003, timestamp=1700000024, base_fee_gwei=35.0)

    block3.transactions.append(Transaction(
        tx_hash="0xcccc1111", sender="0xUser5", receiver="0xUniV3Router",
        value_eth=0.0, gas_price_gwei=36.0, gas_used=130_000, block_position=0,
    ))

    # LIQUIDATION
    block3.transactions.append(Transaction(
        tx_hash="0xLIQ_001", sender="0xLiquidatorBot", receiver="0xAaveLendingPool",
        value_eth=0.0, gas_price_gwei=60.0, gas_used=350_000, block_position=1,
        liquidation=LiquidationCall(
            protocol="Aave V3",
            borrower="0xUnderwater_Borrower",
            debt_token=tokens["DAI"],
            collateral_token=tokens["WETH"],
            debt_repaid=15_000.0,         # Repay 15,000 DAI
            collateral_seized=5.25,       # Receive 5.25 WETH (worth ~$15,750)
            liquidation_bonus_pct=5.0,    # 5% bonus
        ),
    ))

    # Normal tx
    block3.transactions.append(Transaction(
        tx_hash="0xcccc2222", sender="0xUser6", receiver="0xTransfer",
        value_eth=3.0, gas_price_gwei=36.0, gas_used=21_000, block_position=2,
    ))

    blocks.append(block3)

    # =========================================================================
    # Block 4: Contains BOTH a sandwich AND an arbitrage
    # =========================================================================
    block4 = Block(number=18_500_004, timestamp=1700000036, base_fee_gwei=32.0)

    # Sandwich on UNI/WETH pool
    block4.transactions.append(Transaction(
        tx_hash="0xSAND2_FRONT", sender="0xMEVBot_Beta", receiver="0xUniV3Router",
        value_eth=0.0, gas_price_gwei=90.0, gas_used=190_000, block_position=0,
        swaps=[DEXSwap(
            pool_address="0xPool_UNI_WETH",
            token_in=tokens["WETH"], token_out=tokens["UNI"],
            amount_in=5.0, amount_out=625.0,
            effective_price=125.0,
        )],
    ))

    block4.transactions.append(Transaction(
        tx_hash="0xVICTIM_002", sender="0xInnocentUser99", receiver="0xUniV3Router",
        value_eth=0.0, gas_price_gwei=40.0, gas_used=165_000, block_position=1,
        swaps=[DEXSwap(
            pool_address="0xPool_UNI_WETH",
            token_in=tokens["WETH"], token_out=tokens["UNI"],
            amount_in=8.0, amount_out=960.0,
            effective_price=120.0,
        )],
    ))

    block4.transactions.append(Transaction(
        tx_hash="0xSAND2_BACK", sender="0xMEVBot_Beta", receiver="0xUniV3Router",
        value_eth=0.0, gas_price_gwei=88.0, gas_used=185_000, block_position=2,
        swaps=[DEXSwap(
            pool_address="0xPool_UNI_WETH",
            token_in=tokens["UNI"], token_out=tokens["WETH"],
            amount_in=630.0,  # Selling 630 UNI (bought 625, gained 5 from price impact)
            amount_out=5.15,
            effective_price=0.008175,
        )],
    ))

    # Arbitrage in the same block: WETH -> DAI -> USDC -> WETH
    block4.transactions.append(Transaction(
        tx_hash="0xARB_002", sender="0xTriArbBot", receiver="0xArbContract2",
        value_eth=0.0, gas_price_gwei=50.0, gas_used=300_000, block_position=3,
        swaps=[
            DEXSwap(
                pool_address="0xCurve_DAI_USDC",
                token_in=tokens["WETH"], token_out=tokens["DAI"],
                amount_in=20.0, amount_out=60_200.0,
                effective_price=3010.0,
            ),
            DEXSwap(
                pool_address="0xUni_DAI_USDC",
                token_in=tokens["DAI"], token_out=tokens["USDC"],
                amount_in=60_200.0, amount_out=60_350.0,
                effective_price=1.00249,
            ),
            DEXSwap(
                pool_address="0xUni_USDC_WETH",
                token_in=tokens["USDC"], token_out=tokens["WETH"],
                amount_in=60_350.0, amount_out=20.15,
                effective_price=0.000334,
            ),
        ],
    ))

    blocks.append(block4)

    return blocks


# =============================================================================
# Gas Price Analysis
# =============================================================================

def analyze_gas_prices(blocks: list[Block]) -> str:
    """
    Analyze gas price patterns to identify MEV-related transactions.

    MEV transactions typically pay significantly above the block's base fee
    to ensure priority ordering. By comparing each transaction's gas price
    to the block median, we can flag potential MEV activity.

    A "gas premium" of 2x+ above median is a strong MEV signal.
    """
    lines = []
    lines.append("\n--- GAS PRICE ANALYSIS ---")

    for block in blocks:
        if not block.transactions:
            continue

        median_gas = block.median_gas_price
        lines.append(f"\nBlock {block.number}:")
        lines.append(f"  Base fee: {block.base_fee_gwei:.1f} Gwei")
        lines.append(f"  Median gas price: {median_gas:.1f} Gwei")
        lines.append(f"  Transactions with elevated gas (>1.5x median):")

        for tx in block.transactions:
            if tx.gas_price_gwei > median_gas * 1.5:
                premium = tx.gas_price_gwei / median_gas
                lines.append(
                    f"    {tx.tx_hash}: {tx.gas_price_gwei:.1f} Gwei "
                    f"({premium:.1f}x median) — sender: {tx.sender}"
                )

    return "\n".join(lines)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 073: MEV DETECTION ENGINE")
    print("Detecting Sandwich Attacks, Arbitrage, and Liquidations")
    print("=" * 70)

    # Create simulated block data with embedded MEV patterns
    print("\n[1] Generating simulated block data...")
    blocks = create_sample_blocks()
    print(f"    Created {len(blocks)} blocks with {sum(len(b.transactions) for b in blocks)} total transactions")

    for block in blocks:
        print(f"    Block {block.number}: {len(block.transactions)} txs, base fee {block.base_fee_gwei} Gwei")

    # Run MEV detection
    print("\n[2] Running MEV detection engine...")
    detector = MEVDetector(eth_price_usd=3000.0)

    for block in blocks:
        detector.analyze_block(block)
        print(f"    Block {block.number}: "
              f"{len([s for s in detector.sandwiches if s.block_number == block.number])} sandwiches, "
              f"{len([a for a in detector.arbitrages if a.block_number == block.number])} arbs, "
              f"{len([l for l in detector.liquidations if l.block_number == block.number])} liquidations")

    # Generate and print the report
    print("\n[3] Generating report...\n")
    report = detector.generate_report()
    print(report)

    # Gas price analysis
    gas_analysis = analyze_gas_prices(blocks)
    print(gas_analysis)

    # Summary statistics
    print("\n\n--- SUMMARY STATISTICS ---")
    print(f"Blocks analyzed:       {len(blocks)}")
    print(f"Total transactions:    {sum(len(b.transactions) for b in blocks)}")
    print(f"Sandwich attacks:      {len(detector.sandwiches)}")
    print(f"Arbitrage events:      {len(detector.arbitrages)}")
    print(f"Liquidation events:    {len(detector.liquidations)}")

    total_sandwich_profit = sum(s.estimated_profit_usd for s in detector.sandwiches)
    total_arb_profit = sum(a.estimated_profit_usd for a in detector.arbitrages)
    total_liq_bonus = sum(l.bonus_value_usd for l in detector.liquidations)

    print(f"\nSandwich profit:       ${total_sandwich_profit:,.2f}")
    print(f"Arbitrage profit:      ${total_arb_profit:,.2f}")
    print(f"Liquidation bonuses:   ${total_liq_bonus:,.2f}")
    print(f"Total MEV extracted:   ${total_sandwich_profit + total_arb_profit + total_liq_bonus:,.2f}")

    # Show the game theory at play
    print("\n--- KEY INSIGHTS ---")
    print("1. Sandwich attackers pay 2-3x median gas to guarantee ordering")
    print("2. Arbitrage is 'good MEV' — it equalizes prices across venues")
    print("3. Liquidation bots compete on gas price for the right to liquidate")
    print("4. All MEV strategies share a common pattern: information + speed + capital")
    print("5. Flashbots/MEV-Share aims to redistribute this value back to users")
