"""
ISOLATED MARKET DATA TEST 
Tests ONLY the market_data_manager to verify real vs fake data fetching
"""

def test_market_data_isolated():
    import logging
    import sys
    import time
    
    # Force DEBUG logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    print("🔍 ISOLATED MARKET DATA TEST")
    print("="*50)
    
    from market_data_manager import MarketDataManager
    import MetaTrader5 as mt5
    
    # Create manager
    manager = MarketDataManager()
    
    # Test symbols
    symbols = ['EURUSD', 'GBPUSD', 'AUDUSD', 'XAUUSD', 'US30']
    
    print(f"Testing {len(symbols)} symbols with FULL DEBUG logging...")
    print()
    
    total_start = time.time()
    
    for i, symbol in enumerate(symbols, 1):
        print(f"🔄 [{i}/{len(symbols)}] Testing {symbol}...")
        
        # Time the fetch
        start = time.time()
        df = manager.get_live_data(symbol, mt5.TIMEFRAME_M30, 200)
        fetch_time = time.time() - start
        
        # Results
        if df is not None:
            print(f"✅ {symbol}: {len(df)} bars in {fetch_time:.3f}s")
        else:
            print(f"❌ {symbol}: FAILED in {fetch_time:.3f}s")
        
        print("-" * 30)
    
    total_time = time.time() - total_start
    
    print(f"📊 SUMMARY:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average per symbol: {total_time/len(symbols):.3f}s")
    print(f"   Speed: {len(symbols)/total_time:.1f} symbols/second")
    
    if total_time < 1.0:
        print("🚨 SUSPICIOUS: TOO FAST - Likely fake/cached data!")
    else:
        print("✅ TIMING REASONABLE - Real data fetching detected")

if __name__ == "__main__":
    test_market_data_isolated()