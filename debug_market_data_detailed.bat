@echo off
echo 🔍 MARKET DATA DEBUG TEST - DETAILED LOGGING  
echo.
echo Testing REAL vs FAKE market data fetching...
echo.

cd /d "C:\Users\Administrator\Documents\EW-Live_VPS_v2"

echo 📊 Running with DEBUG logging enabled...
python -c "
import logging
import sys

# Set DEBUG logging to see EVERYTHING
logging.basicConfig(
    level=logging.DEBUG, 
    format='%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

print('🚀 Starting DETAILED market data test...')
print()

from market_data_manager import MarketDataManager  
import MetaTrader5 as mt5
import time

market_data = MarketDataManager()

# Test 5 symbols with timing
symbols = ['EURUSD', 'GBPUSD', 'AUDUSD', 'XAUUSD', 'US30']

print('📊 Testing market data fetch speeds...')
print()

total_start = time.time()

for i, symbol in enumerate(symbols, 1):
    print(f'🔄 Testing {symbol} ({i}/{len(symbols)})...')
    
    # Test market data fetch
    start = time.time()
    df = market_data.get_live_data(symbol, mt5.TIMEFRAME_M30, 200)
    fetch_time = time.time() - start
    
    if df is not None:
        print(f'✅ {symbol}: SUCCESS - {len(df)} bars in {fetch_time:.3f}s')
    else:    
        print(f'❌ {symbol}: FAILED in {fetch_time:.3f}s')
    
    print()

total_time = time.time() - total_start
print(f'📈 TOTAL: {len(symbols)} symbols in {total_time:.3f}s')
print(f'🚀 Average: {total_time/len(symbols):.3f}s per symbol')

if total_time < 1.0:
    print('🚨 WARNING: TOO FAST! Likely using cached/fake data!')
else:
    print('✅ TIMING LOOKS REALISTIC for real MT5 data fetch')
"

pause