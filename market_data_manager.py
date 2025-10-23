"""
Market Data Manager - Live MT5 Data Feed
Handles real-time market data acquisition and preprocessing
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

class MarketDataManager:
    """
    Professional market data manager for live MT5 feeds
    Handles multiple symbols, timeframes, and data quality
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mt5_connected = False
        self.symbol_cache = {}
        
    def connect(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                error_code = mt5.last_error()
                self.logger.error(f"MT5 initialization failed: {error_code}")
                return False
            
            # Verify terminal connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.logger.error("MT5 terminal info not available")
                mt5.shutdown()
                return False
            
            # Verify account connection
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("MT5 account info not available")
                mt5.shutdown()
                return False
            
            self.mt5_connected = True
            self.logger.info(f"Connected to MT5: {terminal_info.name} Build {terminal_info.build}")
            self.logger.info(f"Account: {account_info.login} ({account_info.company})")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Close MT5 connection"""
        if self.mt5_connected:
            mt5.shutdown()
            self.mt5_connected = False
            self.logger.info("MT5 connection closed")
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive symbol information"""
        try:
            # Check cache first
            if symbol in self.symbol_cache:
                return self.symbol_cache[symbol]
            
            # Ensure symbol is visible
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.warning(f"Symbol {symbol} not found")
                return None
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    self.logger.warning(f"Failed to select symbol {symbol}")
                    return None
                # Refresh symbol info after selection
                symbol_info = mt5.symbol_info(symbol)
            
            # Extract key information
            info = {
                'name': symbol,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'spread': symbol_info.spread,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'trade_tick_value': symbol_info.trade_tick_value,
                'trade_tick_size': symbol_info.trade_tick_size,
                'contract_size': symbol_info.trade_contract_size,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'path': symbol_info.path
            }
            
            # Cache the information
            self.symbol_cache[symbol] = info
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def get_live_data(self, symbol: str, timeframe: int, bars: int = 200) -> Optional[pd.DataFrame]:
        """
        Get live OHLCV data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'XAUUSD')
            timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_M30)
            bars: Number of historical bars to retrieve
        """
        try:
            if not self.mt5_connected:
                self.logger.error("MT5 not connected")
                return None
            
            # Ensure symbol is available
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return None
            
            # Get historical rates
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) < 50:
                self.logger.warning(f"Insufficient data for {symbol}: {len(rates) if rates is not None else 0} bars")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting live data for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask prices"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time),
                'spread': tick.ask - tick.bid
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add essential technical indicators for Elliott Wave analysis"""
        if len(df) < 50:
            return df
        
        try:
            # ATR (Average True Range) - Critical for Elliott Wave
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
            df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=14).mean()
            df['atr_pct'] = df['atr'] / df['close']
            
            # EMAs for trend filtering
            df['ema_fast'] = df['close'].ewm(span=50).mean()   # 50-period EMA
            df['ema_slow'] = df['close'].ewm(span=200).mean()  # 200-period EMA
            
            # RSI for momentum
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # ADX for trend strength (Elliott Wave filter)
            high = df['high']
            low = df['low']
            close = df['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = minus_dm.abs()
            
            tr = df['tr']
            plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
            minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(14).mean()
            
            # Volume analysis (if available)
            if 'tick_volume' in df.columns:
                df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
            
            # Clean up intermediate columns
            df.drop(['high_low', 'high_close', 'low_close', 'tr'], axis=1, inplace=True)
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate data quality for Elliott Wave analysis"""
        if df is None or len(df) < 100:
            self.logger.warning(f"{symbol}: Insufficient data ({len(df) if df is not None else 0} bars)")
            return False
        
        # Check for missing values
        if df[['open', 'high', 'low', 'close']].isnull().any().any():
            self.logger.warning(f"{symbol}: Missing OHLC data")
            return False
        
        # Check for data consistency
        if not ((df['high'] >= df['low']) & 
                (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) & 
                (df['low'] <= df['close'])).all():
            self.logger.warning(f"{symbol}: Inconsistent OHLC data")
            return False
        
        # Check ATR availability
        if 'atr' not in df.columns or df['atr'].isnull().all():
            self.logger.warning(f"{symbol}: ATR calculation failed")
            return False
        
        return True
    
    def get_multiple_symbols(self, symbols: List[str], timeframe: int, bars: int = 200) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols efficiently"""
        results = {}
        
        for symbol in symbols:
            df = self.get_live_data(symbol, timeframe, bars)
            if df is not None and self.validate_data_quality(df, symbol):
                results[symbol] = df
            else:
                self.logger.warning(f"Skipping {symbol} due to data quality issues")
        
        return results

if __name__ == "__main__":
    # Test the market data manager
    logging.basicConfig(level=logging.INFO)
    
    mdm = MarketDataManager()
    if mdm.connect():
        print("🔌 Market Data Manager Connected")
        
        # Test data retrieval
        symbols = ['EURUSD', 'XAUUSD', 'US30']
        data = mdm.get_multiple_symbols(symbols, mt5.TIMEFRAME_M30)
        
        for symbol, df in data.items():
            print(f"📊 {symbol}: {len(df)} bars, ATR: {df['atr'].iloc[-1]:.5f}")
        
        mdm.disconnect()
    else:
        print("❌ Failed to connect to MT5")