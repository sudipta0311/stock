from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.utils.symbol_resolver import get_symbol_display_name, resolve_nse_symbol


class SymbolResolutionTests(unittest.TestCase):
    def test_tata_motors_maps_to_tmcv(self) -> None:
        self.assertEqual(resolve_nse_symbol("TATAMOTORS"), "TMCV.NS")
        self.assertEqual(get_symbol_display_name("TATAMOTORS"), "Tata Motors CV (TMCV)")


if __name__ == "__main__":
    unittest.main()
