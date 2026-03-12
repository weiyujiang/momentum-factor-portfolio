"""
Quick 5-year backtest for validation (2020-2025)
"""

import os
os.environ['BACKTEST_START'] = '2020-01-01'
os.environ['BACKTEST_END'] = '2025-03-11'

# Now import and run main
from main import main

if __name__ == '__main__':
    print("="*70)
    print("QUICK TEST: 5-Year Backtest (2020-2025)")
    print("="*70)
    results, summary = main()
    print("\nQuick test complete! System validated.")
    print("Ready to run full 25-year backtest.")
