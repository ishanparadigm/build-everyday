"""
Day 073: MEV Detection Script — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py

Tests cover:
- Transaction gas cost calculations
- Block median gas price
- Sandwich attack detection
- Arbitrage detection
- Liquidation detection
- Gas price analysis
- Edge cases (empty blocks, no MEV, unprofitable patterns)
"""

import unittest
from dataclasses import field
from my_solution import (
    Token, DEXSwap, LiquidationCall, Transaction, Block,
    MEVDetector, SandwichDetection, ArbitrageDetection,
    LiquidationDetection, analyze_gas_prices,
)


def make_token(symbol: str) -> Token:
    """Helper to create a token with a dummy address."""
    return Token(symbol=symbol, address=f"0x{symbol}", decimals=18)


WETH = make_token("WETH")
USDC = make_token("USDC")
DAI = make_token("DAI")
LINK = make_token("LINK")
UNI = make_token("UNI")


class TestTransactionGasCost(unittest.TestCase):
    """Test gas cost calculations on Transaction."""

    def test_gas_cost_eth(self):
        tx = Transaction(
            tx_hash="0x1", sender="0xA", receiver="0xB",
            value_eth=0.0, gas_price_gwei=50.0, gas_used=200_000,
            block_position=0,
        )
        # 50 Gwei * 200,000 gas = 10,000,000 Gwei = 0.01 ETH
        self.assertAlmostEqual(tx.gas_cost_eth, 0.01, places=6)

    def test_gas_cost_usd(self):
        tx = Transaction(
            tx_hash="0x1", sender="0xA", receiver="0xB",
            value_eth=0.0, gas_price_gwei=50.0, gas_used=200_000,
            block_position=0,
        )
        # 0.01 ETH * $3000 = $30
        self.assertAlmostEqual(tx.gas_cost_usd, 30.0, places=2)


class TestBlockMedianGas(unittest.TestCase):
    """Test median gas price calculation."""

    def test_median_gas_price(self):
        block = Block(number=1, timestamp=0, base_fee_gwei=20.0)
        for i, gas in enumerate([20.0, 30.0, 50.0, 25.0, 35.0]):
            block.transactions.append(Transaction(
                tx_hash=f"0x{i}", sender="0xA", receiver="0xB",
                value_eth=0.0, gas_price_gwei=gas, gas_used=21000,
                block_position=i,
            ))
        self.assertAlmostEqual(block.median_gas_price, 30.0, places=1)

    def test_empty_block(self):
        block = Block(number=1, timestamp=0, base_fee_gwei=20.0)
        self.assertEqual(block.median_gas_price, 0.0)


class TestSandwichDetection(unittest.TestCase):
    """Test sandwich attack detection."""

    def _make_sandwich_block(self) -> Block:
        """Create a block with a clear sandwich pattern."""
        block = Block(number=100, timestamp=0, base_fee_gwei=30.0)

        # Front-run: attacker buys WETH with USDC
        block.transactions.append(Transaction(
            tx_hash="0xFRONT", sender="0xAttacker", receiver="0xRouter",
            value_eth=0.0, gas_price_gwei=80.0, gas_used=180_000,
            block_position=0,
            swaps=[DEXSwap(
                pool_address="0xPool1",
                token_in=USDC, token_out=WETH,
                amount_in=30_000.0, amount_out=10.0,
                effective_price=0.000333,
            )],
        ))

        # Victim: also buys WETH with USDC
        block.transactions.append(Transaction(
            tx_hash="0xVICTIM", sender="0xVictim", receiver="0xRouter",
            value_eth=0.0, gas_price_gwei=35.0, gas_used=160_000,
            block_position=1,
            swaps=[DEXSwap(
                pool_address="0xPool1",
                token_in=USDC, token_out=WETH,
                amount_in=5_000.0, amount_out=1.6,
                effective_price=0.00032,
            )],
        ))

        # Back-run: attacker sells WETH for USDC
        block.transactions.append(Transaction(
            tx_hash="0xBACK", sender="0xAttacker", receiver="0xRouter",
            value_eth=0.0, gas_price_gwei=78.0, gas_used=175_000,
            block_position=2,
            swaps=[DEXSwap(
                pool_address="0xPool1",
                token_in=WETH, token_out=USDC,
                amount_in=10.1, amount_out=30_500.0,  # Sold more than bought
                effective_price=3019.80,
            )],
        ))

        return block

    def test_detects_sandwich(self):
        block = self._make_sandwich_block()
        detector = MEVDetector(eth_price_usd=3000.0)
        detector.analyze_block(block)
        self.assertEqual(len(detector.sandwiches), 1)

    def test_sandwich_attacker(self):
        block = self._make_sandwich_block()
        detector = MEVDetector(eth_price_usd=3000.0)
        detector.analyze_block(block)
        self.assertEqual(detector.sandwiches[0].attacker, "0xAttacker")
        self.assertEqual(detector.sandwiches[0].victim, "0xVictim")

    def test_sandwich_profit_positive(self):
        block = self._make_sandwich_block()
        detector = MEVDetector(eth_price_usd=3000.0)
        detector.analyze_block(block)
        self.assertGreater(detector.sandwiches[0].profit_token_amount, 0)

    def test_no_sandwich_same_sender(self):
        """If all three txs are from the same sender, it's not a sandwich."""
        block = Block(number=200, timestamp=0, base_fee_gwei=30.0)
        for i, (amt_in, amt_out, tok_in, tok_out) in enumerate([
            (30000, 10.0, USDC, WETH),
            (5000, 1.6, USDC, WETH),
            (10.1, 30500, WETH, USDC),
        ]):
            block.transactions.append(Transaction(
                tx_hash=f"0x{i}", sender="0xSameSender", receiver="0xR",
                value_eth=0.0, gas_price_gwei=50.0, gas_used=150_000,
                block_position=i,
                swaps=[DEXSwap("0xPool1", tok_in, tok_out, amt_in, amt_out, 1.0)],
            ))
        detector = MEVDetector()
        detector.analyze_block(block)
        self.assertEqual(len(detector.sandwiches), 0)


class TestArbitrageDetection(unittest.TestCase):
    """Test arbitrage detection."""

    def _make_arb_block(self) -> Block:
        """Create a block with a circular arbitrage."""
        block = Block(number=300, timestamp=0, base_fee_gwei=28.0)
        block.transactions.append(Transaction(
            tx_hash="0xARB", sender="0xSearcher", receiver="0xArbContract",
            value_eth=0.0, gas_price_gwei=55.0, gas_used=250_000,
            block_position=0,
            swaps=[
                DEXSwap("0xPoolA", WETH, LINK, 10.0, 2000.0, 200.0),
                DEXSwap("0xPoolB", LINK, WETH, 2000.0, 10.15, 0.005075),
            ],
        ))
        return block

    def test_detects_arbitrage(self):
        block = self._make_arb_block()
        detector = MEVDetector(eth_price_usd=3000.0)
        detector.analyze_block(block)
        self.assertEqual(len(detector.arbitrages), 1)

    def test_arb_profit(self):
        block = self._make_arb_block()
        detector = MEVDetector(eth_price_usd=3000.0)
        detector.analyze_block(block)
        arb = detector.arbitrages[0]
        self.assertAlmostEqual(arb.profit_amount, 0.15, places=2)
        self.assertEqual(arb.profit_token, "WETH")

    def test_no_arb_non_circular(self):
        """Non-circular path should not be detected as arbitrage."""
        block = Block(number=301, timestamp=0, base_fee_gwei=28.0)
        block.transactions.append(Transaction(
            tx_hash="0xNOARB", sender="0xUser", receiver="0xRouter",
            value_eth=0.0, gas_price_gwei=30.0, gas_used=200_000,
            block_position=0,
            swaps=[
                DEXSwap("0xP1", WETH, USDC, 1.0, 3000.0, 3000.0),
                DEXSwap("0xP2", USDC, DAI, 3000.0, 3005.0, 1.00167),
            ],
        ))
        detector = MEVDetector()
        detector.analyze_block(block)
        self.assertEqual(len(detector.arbitrages), 0)

    def test_no_arb_single_swap(self):
        """Single swap is not arbitrage."""
        block = Block(number=302, timestamp=0, base_fee_gwei=28.0)
        block.transactions.append(Transaction(
            tx_hash="0xSINGLE", sender="0xUser", receiver="0xRouter",
            value_eth=0.0, gas_price_gwei=30.0, gas_used=150_000,
            block_position=0,
            swaps=[DEXSwap("0xP1", WETH, USDC, 1.0, 3000.0, 3000.0)],
        ))
        detector = MEVDetector()
        detector.analyze_block(block)
        self.assertEqual(len(detector.arbitrages), 0)


class TestLiquidationDetection(unittest.TestCase):
    """Test liquidation detection."""

    def _make_liq_block(self) -> Block:
        block = Block(number=400, timestamp=0, base_fee_gwei=35.0)
        block.transactions.append(Transaction(
            tx_hash="0xLIQ", sender="0xLiquidator", receiver="0xAave",
            value_eth=0.0, gas_price_gwei=60.0, gas_used=350_000,
            block_position=0,
            liquidation=LiquidationCall(
                protocol="Aave V3",
                borrower="0xBorrower",
                debt_token=DAI,
                collateral_token=WETH,
                debt_repaid=15_000.0,
                collateral_seized=5.25,
                liquidation_bonus_pct=5.0,
            ),
        ))
        return block

    def test_detects_liquidation(self):
        block = self._make_liq_block()
        detector = MEVDetector(eth_price_usd=3000.0)
        detector.analyze_block(block)
        self.assertEqual(len(detector.liquidations), 1)

    def test_liquidation_details(self):
        block = self._make_liq_block()
        detector = MEVDetector(eth_price_usd=3000.0)
        detector.analyze_block(block)
        liq = detector.liquidations[0]
        self.assertEqual(liq.liquidator, "0xLiquidator")
        self.assertEqual(liq.borrower, "0xBorrower")
        self.assertEqual(liq.protocol, "Aave V3")
        self.assertAlmostEqual(liq.bonus_pct, 5.0)

    def test_no_liquidation_in_normal_tx(self):
        block = Block(number=401, timestamp=0, base_fee_gwei=30.0)
        block.transactions.append(Transaction(
            tx_hash="0xNORMAL", sender="0xUser", receiver="0xRouter",
            value_eth=1.0, gas_price_gwei=30.0, gas_used=21_000,
            block_position=0,
        ))
        detector = MEVDetector()
        detector.analyze_block(block)
        self.assertEqual(len(detector.liquidations), 0)


class TestGasAnalysis(unittest.TestCase):
    """Test gas price analysis function."""

    def test_identifies_elevated_gas(self):
        block = Block(number=500, timestamp=0, base_fee_gwei=30.0)
        # 5 normal txs at ~30 Gwei, 1 MEV tx at 90 Gwei
        for i in range(5):
            block.transactions.append(Transaction(
                tx_hash=f"0xN{i}", sender=f"0xUser{i}", receiver="0xR",
                value_eth=0.0, gas_price_gwei=30.0 + i, gas_used=21_000,
                block_position=i,
            ))
        block.transactions.append(Transaction(
            tx_hash="0xMEV", sender="0xBot", receiver="0xR",
            value_eth=0.0, gas_price_gwei=90.0, gas_used=200_000,
            block_position=5,
        ))
        result = analyze_gas_prices([block])
        self.assertIn("0xMEV", result)
        self.assertIn("0xBot", result)


if __name__ == "__main__":
    unittest.main()
