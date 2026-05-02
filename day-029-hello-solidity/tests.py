"""
Day 29: Hello World Solidity Contract — Test Suite

Tests the smart contract VM simulation for correctness:
storage, gas metering, function dispatch, events, access control, and reverts.

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
from my_solution import (
    ContractStorage, GasMeter, OutOfGasError, EventLog, Event,
    FunctionRegistry, compute_selector, MessageContext,
    SmartContract, HelloWorldContract, MiniEVM, RevertError,
    EVENT_LOG_GAS,
)


class TestGasMeter(unittest.TestCase):
    """Test gas metering and out-of-gas behavior."""

    def test_basic_consumption(self):
        meter = GasMeter(10000)
        meter.consume(5000, "op1")
        self.assertEqual(meter.gas_used, 5000)
        self.assertEqual(meter.gas_remaining, 5000)

    def test_out_of_gas_raises(self):
        meter = GasMeter(1000)
        with self.assertRaises(OutOfGasError):
            meter.consume(1500, "expensive_op")

    def test_cumulative_gas(self):
        meter = GasMeter(10000)
        meter.consume(3000)
        meter.consume(3000)
        meter.consume(3000)
        self.assertEqual(meter.gas_used, 9000)
        # This should push over the limit
        with self.assertRaises(OutOfGasError):
            meter.consume(2000)


class TestContractStorage(unittest.TestCase):
    """Test storage read/write with gas accounting."""

    def test_sstore_and_sload(self):
        storage = ContractStorage()
        meter = GasMeter(100000)
        storage.sstore("key", "value", meter)
        result = storage.sload("key", meter)
        self.assertEqual(result, "value")

    def test_cold_vs_warm_gas(self):
        storage = ContractStorage()
        meter = GasMeter(100000)
        storage.sstore("key", "value", meter)
        gas_before = meter.gas_used
        # First read (cold access in sload)
        storage.sload("key", meter)
        gas_after_warm = meter.gas_used  # warm because sstore already warmed it
        # Should be warm (100 gas) since sstore warmed the slot
        self.assertEqual(gas_after_warm - gas_before, ContractStorage.SLOAD_WARM_GAS)

    def test_unset_slot_returns_none(self):
        storage = ContractStorage()
        meter = GasMeter(100000)
        result = storage.sload("nonexistent", meter)
        self.assertIsNone(result)

    def test_snapshot_and_restore(self):
        storage = ContractStorage()
        meter = GasMeter(100000)
        storage.sstore("key", "original", meter)
        snap = storage.snapshot()
        storage.sstore("key", "modified", meter)
        self.assertEqual(storage.sload("key", meter), "modified")
        storage.restore(snap)
        self.assertEqual(storage.sload("key", meter), "original")

    def test_new_slot_costs_more(self):
        storage = ContractStorage()
        meter1 = GasMeter(100000)
        storage.sstore("new_key", "value", meter1)
        new_cost = meter1.gas_used

        meter2 = GasMeter(100000)
        storage.sstore("new_key", "updated", meter2)
        update_cost = meter2.gas_used

        self.assertGreater(new_cost, update_cost)


class TestFunctionDispatch(unittest.TestCase):
    """Test function selector computation and dispatch."""

    def test_selector_deterministic(self):
        sel1 = compute_selector("getGreeting()")
        sel2 = compute_selector("getGreeting()")
        self.assertEqual(sel1, sel2)

    def test_different_signatures_different_selectors(self):
        sel1 = compute_selector("getGreeting()")
        sel2 = compute_selector("setGreeting(string)")
        self.assertNotEqual(sel1, sel2)

    def test_selector_format(self):
        sel = compute_selector("transfer(address,uint256)")
        self.assertTrue(sel.startswith("0x"))
        self.assertEqual(len(sel), 10)  # "0x" + 8 hex chars

    def test_register_and_dispatch(self):
        registry = FunctionRegistry()
        handler = lambda msg, gas: "hello"
        selector = registry.register("greet", "greet()", handler)
        fn = registry.dispatch(selector)
        self.assertIsNotNone(fn)
        self.assertEqual(fn.name, "greet")

    def test_resolve_by_name(self):
        registry = FunctionRegistry()
        handler = lambda msg, gas: "hello"
        registry.register("greet", "greet()", handler)
        fn = registry.resolve_by_name("greet")
        self.assertIsNotNone(fn)
        self.assertEqual(fn.signature, "greet()")


class TestEventLog(unittest.TestCase):
    """Test event emission and filtering."""

    def test_emit_and_retrieve(self):
        log = EventLog()
        meter = GasMeter(100000)
        event = Event(name="Transfer", args={"from": "a", "to": "b"}, emitter="0x1")
        log.emit(event, meter)
        events = log.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "Transfer")

    def test_filter_by_name(self):
        log = EventLog()
        meter = GasMeter(100000)
        log.emit(Event(name="Transfer", args={}, emitter="0x1"), meter)
        log.emit(Event(name="Approval", args={}, emitter="0x1"), meter)
        log.emit(Event(name="Transfer", args={}, emitter="0x1"), meter)
        transfers = log.get_events("Transfer")
        self.assertEqual(len(transfers), 2)

    def test_emit_charges_gas(self):
        log = EventLog()
        meter = GasMeter(100000)
        log.emit(Event(name="Test", args={}, emitter="0x1"), meter)
        self.assertEqual(meter.gas_used, EVENT_LOG_GAS)


class TestHelloWorldContract(unittest.TestCase):
    """End-to-end contract tests via the MiniEVM."""

    def setUp(self):
        self.evm = MiniEVM()
        self.alice = "0xAlice"
        self.bob = "0xBob"
        self.contract, self.receipt = self.evm.deploy_contract(
            self.alice, "Hello!", gas_limit=300000
        )
        self.addr = self.receipt["contract_address"]

    def test_deployment_succeeds(self):
        self.assertIsNotNone(self.contract)
        self.assertEqual(self.receipt["status"], "success")

    def test_get_greeting(self):
        result = self.evm.call_function(self.addr, "getGreeting", sender=self.alice)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "Hello!")

    def test_set_greeting_by_owner(self):
        result = self.evm.call_function(
            self.addr, "setGreeting", sender=self.alice, new_greeting="Updated!"
        )
        self.assertEqual(result["status"], "success")
        result = self.evm.call_function(self.addr, "getGreeting", sender=self.alice)
        self.assertEqual(result["result"], "Updated!")

    def test_set_greeting_by_non_owner_reverts(self):
        result = self.evm.call_function(
            self.addr, "setGreeting", sender=self.bob, new_greeting="Hacked!"
        )
        self.assertEqual(result["status"], "reverted")
        # Verify greeting unchanged
        result = self.evm.call_function(self.addr, "getGreeting", sender=self.alice)
        self.assertEqual(result["result"], "Hello!")

    def test_transfer_ownership(self):
        # Transfer to Bob
        result = self.evm.call_function(
            self.addr, "transferOwnership", sender=self.alice, new_owner=self.bob
        )
        self.assertEqual(result["status"], "success")
        # Bob can now set greeting
        result = self.evm.call_function(
            self.addr, "setGreeting", sender=self.bob, new_greeting="Bob's!"
        )
        self.assertEqual(result["status"], "success")
        # Alice can no longer set greeting
        result = self.evm.call_function(
            self.addr, "setGreeting", sender=self.alice, new_greeting="Alice tries"
        )
        self.assertEqual(result["status"], "reverted")

    def test_greeting_count_increments(self):
        self.evm.call_function(self.addr, "setGreeting", sender=self.alice,
                               new_greeting="One")
        self.evm.call_function(self.addr, "setGreeting", sender=self.alice,
                               new_greeting="Two")
        result = self.evm.call_function(self.addr, "getGreetingCount", sender=self.alice)
        self.assertEqual(result["result"], 2)

    def test_out_of_gas_reverts(self):
        result = self.evm.call_function(
            self.addr, "setGreeting", sender=self.alice,
            new_greeting="Fail", gas_limit=21500
        )
        self.assertIn("out of gas", result["status"])
        # Greeting should be unchanged
        result = self.evm.call_function(self.addr, "getGreeting", sender=self.alice)
        self.assertEqual(result["result"], "Hello!")

    def test_events_emitted_on_greeting_change(self):
        self.evm.call_function(
            self.addr, "setGreeting", sender=self.alice, new_greeting="New!"
        )
        events = self.contract.event_log.get_events("GreetingChanged")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].args["oldGreeting"], "Hello!")
        self.assertEqual(events[0].args["newGreeting"], "New!")

    def test_unknown_function_returns_error(self):
        result = self.evm.call_function(self.addr, "nonexistent", sender=self.alice)
        self.assertEqual(result["status"], "error")

    def test_call_nonexistent_contract(self):
        result = self.evm.call_function("0xDEAD", "getGreeting", sender=self.alice)
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
