"""
Dynamic Price Validation Module
Real-time broker requirements validation for Elliott Wave trades
Automatically queries MT5 for symbol specifications and validates orders
"""

import MetaTrader5 as mt5
import logging
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime

class SymbolInfo(NamedTuple):
    """Symbol specification from broker"""
    symbol: str
    digits: int
    point: float
    spread: int
    stops_level: int
    trade_mode: int
    min_volume: float
    max_volume: float
    volume_step: float

@dataclass
class ValidationResult:
    """Result of price validation"""
    is_valid: bool
    adjusted_sl: Optional[float] = None
    adjusted_tp: Optional[float] = None
    error_message: Optional[str] = None
    pip_distance_sl: Optional[float] = None
    pip_distance_tp: Optional[float] = None

class DynamicPriceValidator:
    """
    Real-time price validator that queries live broker requirements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._symbol_cache: Dict[str, SymbolInfo] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self.cache_duration_minutes = 5  # Refresh symbol info every 5 minutes
        
    def get_symbol_info(self, symbol: str, force_refresh: bool = False) -> Optional[SymbolInfo]:
        """Get real-time symbol information from MT5"""
        
        # Check cache first (unless force refresh)
        if not force_refresh and symbol in self._symbol_cache:
            cache_time = self._cache_timestamp.get(symbol, datetime.min)
            if (datetime.now() - cache_time).total_seconds() < self.cache_duration_minutes * 60:
                return self._symbol_cache[symbol]
        
        try:
            # Get fresh symbol info from MT5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"❌ Symbol {symbol} not found in MT5")
                return None
                
            # Get current spread
            tick = mt5.symbol_info_tick(symbol)
            current_spread = 0
            if tick:
                current_spread = int((tick.ask - tick.bid) / symbol_info.point)
            
            # Create SymbolInfo object
            info = SymbolInfo(
                symbol=symbol,
                digits=symbol_info.digits,
                point=symbol_info.point,
                spread=current_spread,
                stops_level=symbol_info.trade_stops_level,
                trade_mode=symbol_info.trade_mode,
                min_volume=symbol_info.volume_min,
                max_volume=symbol_info.volume_max,
                volume_step=symbol_info.volume_step
            )
            
            # Cache the result
            self._symbol_cache[symbol] = info
            self._cache_timestamp[symbol] = datetime.now()
            
            self.logger.info(f"📊 {symbol} Info: digits={info.digits}, point={info.point}, "
                           f"spread={info.spread}, stops_level={info.stops_level}")
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Error getting symbol info for {symbol}: {e}")
            return None
    
    def calculate_pip_distance(self, price1: float, price2: float, symbol_info: SymbolInfo) -> float:
        """Calculate pip distance using real symbol specifications"""
        
        price_diff = abs(price1 - price2)
        
        # Handle different symbol types correctly
        if symbol_info.symbol.startswith('US') or symbol_info.symbol in ['NAS100', 'UK100', 'DE40']:
            # Indices: 1 pip = 1 point (regardless of digits)
            pip_distance = price_diff
        elif 'JPY' in symbol_info.symbol:
            # JPY pairs: usually 3 digits, 1 pip = 0.01
            pip_distance = price_diff * 100
        elif symbol_info.symbol in ['XAUUSD', 'XAGUSD']:
            # Metals: usually 2-3 digits, 1 pip = 0.1
            pip_distance = price_diff * 10
        else:
            # Currency pairs: handle 4-digit vs 5-digit
            if symbol_info.digits == 5:
                # 5-digit pairs: 1 pip = 0.0001 (10 points)
                pip_distance = price_diff * 10000
            else:
                # 4-digit pairs: 1 pip = 0.0001
                pip_distance = price_diff * 10000
            
        self.logger.debug(f"{symbol_info.symbol}: Price diff={price_diff:.5f}, "
                         f"digits={symbol_info.digits}, distance={pip_distance:.1f} pips")
        
        return pip_distance
    
    def get_minimum_distance(self, symbol_info: SymbolInfo) -> float:
        """Calculate minimum stop/TP distance in pips based on broker requirements"""
        
        # Conservative minimums per symbol type (in pips)
        conservative_minimums = {
            'JPY': 20,    # JPY pairs
            'METAL': 50,  # Gold/Silver  
            'INDEX': 100, # Indices
            'MAJOR': 15,  # Major pairs
            'MINOR': 25,  # Minor pairs
            'EXOTIC': 35  # Exotic pairs
        }
        
        symbol_type = self._get_symbol_type(symbol_info.symbol)
        conservative_min = conservative_minimums.get(symbol_type, 20)
        
        # Broker minimum (stops_level is usually in points, not pips)
        broker_minimum_points = symbol_info.stops_level
        spread_points = symbol_info.spread
        
        # Convert broker minimums to pips based on symbol type
        if symbol_type == 'INDEX':
            # For indices, points = pips usually
            broker_minimum_pips = broker_minimum_points + spread_points + 10
        elif symbol_type == 'METAL':
            # For metals, usually need more buffer
            broker_minimum_pips = max(broker_minimum_points / 10, spread_points + 20)
        else:
            # For currency pairs
            if symbol_info.digits == 5:
                # 5-digit: 10 points = 1 pip
                broker_minimum_pips = (broker_minimum_points + spread_points + 10) / 10
            else:
                # 4-digit: 1 point = 1 pip  
                broker_minimum_pips = broker_minimum_points + spread_points + 5
        
        final_minimum = max(broker_minimum_pips, conservative_min)
        
        self.logger.debug(f"{symbol_info.symbol}: broker_min={broker_minimum_points}pts, "
                         f"spread={spread_points}pts, type={symbol_type}, "
                         f"final_min={final_minimum:.1f} pips")
        
        return final_minimum
    
    def _get_symbol_type(self, symbol: str) -> str:
        """Determine symbol type for minimum distance calculation"""
        if 'JPY' in symbol:
            return 'JPY'
        elif symbol in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER']:
            return 'METAL'
        elif symbol in ['US30', 'NAS100', 'UK100', 'DE40', 'FR40', 'AUS200']:
            return 'INDEX'
        elif symbol in ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCHF', 'USDCAD', 'USDJPY']:
            return 'MAJOR'
        elif any(pair in symbol for pair in ['EUR', 'GBP', 'AUD', 'USD', 'CHF', 'CAD']):
            return 'MINOR'
        else:
            return 'EXOTIC'
    
    def validate_order(self, symbol: str, entry_price: float, 
                      stop_loss: float, take_profit: float) -> ValidationResult:
        """
        Validate complete order against real broker requirements
        Automatically adjusts invalid parameters
        """
        
        # Get live symbol information
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return ValidationResult(
                is_valid=False,
                error_message=f"Cannot get symbol info for {symbol}"
            )
        
        # Check if symbol is tradeable
        if symbol_info.trade_mode == 0:  # TRADE_MODE_DISABLED
            return ValidationResult(
                is_valid=False,
                error_message=f"{symbol} trading is disabled"
            )
        
        # Calculate current distances
        sl_distance = self.calculate_pip_distance(entry_price, stop_loss, symbol_info)
        tp_distance = self.calculate_pip_distance(entry_price, take_profit, symbol_info)
        
        # Get minimum required distance
        min_distance = self.get_minimum_distance(symbol_info)
        
        # Initialize adjustment variables
        adjusted_sl = stop_loss
        adjusted_tp = take_profit
        is_valid = True
        
        # Validate and adjust stop loss
        if sl_distance < min_distance:
            self.logger.warning(f"🔧 {symbol}: SL too close - {sl_distance:.1f} pips, "
                              f"adjusting to {min_distance:.1f} pips")
            
            # Adjust stop loss to minimum distance
            adjusted_sl = self._adjust_price_by_pips(
                entry_price, min_distance, symbol_info, 
                is_stop_loss=True, is_long=(stop_loss < entry_price)
            )
        
        # Validate and adjust take profit
        min_tp_distance = min_distance
        if tp_distance < min_tp_distance:
            self.logger.warning(f"🔧 {symbol}: TP too close - {tp_distance:.1f} pips, "
                              f"adjusting to {min_tp_distance:.1f} pips")
            
            # Adjust take profit to minimum distance
            adjusted_tp = self._adjust_price_by_pips(
                entry_price, min_tp_distance, symbol_info,
                is_stop_loss=False, is_long=(take_profit > entry_price)
            )
        
        # Recalculate final distances
        final_sl_distance = self.calculate_pip_distance(entry_price, adjusted_sl, symbol_info)
        final_tp_distance = self.calculate_pip_distance(entry_price, adjusted_tp, symbol_info)
        
        self.logger.info(f"✅ {symbol}: Validation complete - "
                        f"SL: {final_sl_distance:.1f} pips, TP: {final_tp_distance:.1f} pips")
        
        return ValidationResult(
            is_valid=is_valid,
            adjusted_sl=adjusted_sl,
            adjusted_tp=adjusted_tp,
            pip_distance_sl=final_sl_distance,
            pip_distance_tp=final_tp_distance
        )

    def get_current_price(self, symbol: str) -> Optional[Tuple[float, float]]:
        """Get current bid/ask prices for symbol"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return (tick.bid, tick.ask)
            return None
        except Exception as e:
            self.logger.error(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def format_price(self, price: float, symbol_info: SymbolInfo) -> float:
        """Format price to correct decimal places for symbol"""
        return round(price, symbol_info.digits)
    
    def _adjust_price_by_pips(self, entry_price: float, pip_distance: float, 
                             symbol_info: SymbolInfo, is_stop_loss: bool, is_long: bool) -> float:
        """Adjust price by specified pip distance based on symbol type"""
        
        # Calculate price change based on symbol type
        if symbol_info.symbol.startswith('US') or symbol_info.symbol in ['NAS100', 'UK100', 'DE40']:
            # Indices: 1 pip = 1 point
            price_change = pip_distance
        elif 'JPY' in symbol_info.symbol:
            # JPY pairs: 1 pip = 0.01
            price_change = pip_distance * 0.01
        elif symbol_info.symbol in ['XAUUSD', 'XAGUSD']:
            # Metals: 1 pip = 0.1
            price_change = pip_distance * 0.1
        else:
            # Currency pairs: 1 pip = 0.0001
            price_change = pip_distance * 0.0001
        
        # Apply direction based on position type and SL/TP
        if is_stop_loss:
            # Stop loss is opposite direction to position
            if is_long:
                # Long position: SL below entry
                adjusted_price = entry_price - price_change
            else:
                # Short position: SL above entry
                adjusted_price = entry_price + price_change
        else:
            # Take profit is same direction as position
            if is_long:
                # Long position: TP above entry
                adjusted_price = entry_price + price_change
            else:
                # Short position: TP below entry
                adjusted_price = entry_price - price_change
        
        # Format to correct decimal places
        return self.format_price(adjusted_price, symbol_info)