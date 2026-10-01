"""Demonstrate two known bugs are detected, without changing source files."""

import io
import unittest
from unittest.mock import patch

from hybrid_contract import HybridResources
from test_hybrid_contract import HybridContractTests


def wrong_batch_order(self, request_ids):
    # Incorrect: sorts by physical slot instead of respecting requested order.
    return [tuple(self.states[self.requests[r].slot]['recurrent'])
            for r in sorted(request_ids, key=lambda r: self.requests[r].slot)]


def main():
    cases = [
        ('_reset_slot', lambda self, slot: None,
         'test_reused_slot_clears_both_states_not_only_counter'),
        ('observe_batch', wrong_batch_order,
         'test_batch_reorder_preserves_request_identity'),
    ]
    for method, broken, test in cases:
        output = io.StringIO()
        with patch.object(HybridResources, method, broken):
            result = unittest.TextTestRunner(stream=output).run(
                unittest.TestSuite([HybridContractTests(test)]))
        if len(result.failures) != 1 or result.errors:
            raise AssertionError('Mutation was not caught as expected: ' + output.getvalue())
        print('EXPECTED FAILURE detected:', method)
    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(HybridContractTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print('Original implementation restored; all 12 tests pass.')


if __name__ == '__main__':
    main()
