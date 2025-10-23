"""
Trade Executor - Professional MT5 Order Management
Handles order execution, position management, and monitoring
Integrated with dynamic price validation for broker-specific requirements
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum
import logging
from datetime import datetime, timedelta
import time

from signal_generator import TradingSignal, SignalType
from risk_manager import PositionSize
from price_validator import DynamicPriceValidator, ValidationResult

class OrderType(Enum):
    """MT5 order types"""
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP

class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"

@dataclass
class ExecutionResult:
    """Order execution result"""
    success: bool
    order_id: Optional[int]
    position_id: Optional[int]
    price: Optional[float]
    volume: float
    error_code: int
    error_message: str
    execution_time: datetime
    slippage_pips: float = 0.0

@dataclass
class Position:
    """Active position tracking"""
    position_id: int
    symbol: str
    type: str  # 'buy' or 'sell'
    volume: float
    open_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    swap: float
    profit: float
    comment: str
    open_time: datetime
    
    # Elliott Wave context
    wave_pattern: str
    signal_confidence: float
    
    def update_current_price(self, price: float):
        """Update current price and recalculate profit"""
        self.current_price = price
        if self.type == 'buy':
            self.profit = (price - self.open_price) * self.volume * 100000  # Simplified
        else:
            self.profit = (self.open_price - price) * self.volume * 100000

class TradeExecutor:
    """
    Professional trade execution engine
    Handles all MT5 order management and position tracking
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mt5_connected = False
        self.active_positions = {}  # position_id -> Position
        self.pending_orders = {}    # order_id -> order_info
        
        # Initialize dynamic price validator
        self.price_validator = DynamicPriceValidator()
        
        # Execution settings
        self.max_slippage = 3.0     # Maximum slippage in pips
        self.retry_attempts = 3     # Number of retry attempts
        self.retry_delay = 1.0      # Delay between retries (seconds)
        
        # Magic number for Elliott Wave EA
        self.magic_number = 202501  # Elliott Wave 2025-01
    
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            if not mt5.initialize():
                error = mt5.last_error()
                self.logger.error(f"MT5 initialization failed: {error}")
                return False
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                mt5.shutdown()
                return False
            
            self.mt5_connected = True
            self.logger.info(f"Trade Executor connected to account {account_info.login}")
            
            # Load existing positions
            self._load_existing_positions()
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.mt5_connected:
            mt5.shutdown()
            self.mt5_connected = False
            self.logger.info("Trade Executor disconnected")
    
    def get_account_balance(self) -> Optional[float]:
        """Get current account balance"""
        try:
            if not self.mt5_connected:
                return None
            
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            return float(account_info.balance)
            
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return None
    
    def get_account_info(self) -> Optional[Dict]:
        """Get comprehensive account information"""
        try:
            if not self.mt5_connected:
                return None
            
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            return {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'currency': account_info.currency,
                'company': account_info.company,
                'leverage': account_info.leverage
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def execute_signal(self, signal: TradingSignal, position_size: PositionSize) -> ExecutionResult:
        """
        Execute trading signal with automatic price validation
        
        Args:
            signal: Trading signal from signal generator
            position_size: Position size from risk manager
        """
        try:
            if not self.mt5_connected:
                return ExecutionResult(
                    success=False, order_id=None, position_id=None, price=None,
                    volume=0, error_code=-1, error_message="MT5 not connected",
                    execution_time=datetime.now()
                )

            if not position_size.is_valid:
                return ExecutionResult(
                    success=False, order_id=None, position_id=None, price=None,
                    volume=0, error_code=-1, error_message=f"Invalid position size: {position_size.reason}",
                    execution_time=datetime.now()
                )
            
            # CHECK FOR EXISTING POSITIONS - Prevent duplicates
            if self._has_existing_position(signal.symbol):
                return ExecutionResult(
                    success=False, order_id=None, position_id=None, price=None,
                    volume=0, error_code=-2, error_message=f"Position already exists for {signal.symbol}",
                    execution_time=datetime.now()
                )            # DYNAMIC PRICE VALIDATION - Auto-fix order parameters
            self.logger.info(f"🔍 Validating {signal.symbol} order parameters...")
            validation = self.price_validator.validate_order(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if not validation.is_valid:
                return ExecutionResult(
                    success=False, order_id=None, position_id=None, price=None,
                    volume=0, error_code=-1, error_message=f"Price validation failed: {validation.error_message}",
                    execution_time=datetime.now()
                )
            
            # Use validated/adjusted prices
            validated_sl = validation.adjusted_sl
            validated_tp = validation.adjusted_tp
            
            # Log adjustments if any were made
            if validated_sl != signal.stop_loss:
                self.logger.info(f"🔧 {signal.symbol}: Stop Loss adjusted from {signal.stop_loss:.5f} to {validated_sl:.5f} "
                               f"({validation.pip_distance_sl:.1f} pips)")
            
            if validated_tp != signal.take_profit:
                self.logger.info(f"🔧 {signal.symbol}: Take Profit adjusted from {signal.take_profit:.5f} to {validated_tp:.5f} "
                               f"({validation.pip_distance_tp:.1f} pips)")
            
            # Prepare order request
            if signal.signal_type == SignalType.BUY:
                order_type = mt5.ORDER_TYPE_BUY
                price = signal.entry_price
            elif signal.signal_type == SignalType.SELL:
                order_type = mt5.ORDER_TYPE_SELL
                price = signal.entry_price
            else:
                return ExecutionResult(
                    success=False, order_id=None, position_id=None, price=None,
                    volume=0, error_code=-1, error_message=f"Unsupported signal type: {signal.signal_type}",
                    execution_time=datetime.now()
                )
            
            # Create order request with validated prices
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": position_size.lot_size,
                "type": order_type,
                "price": price,
                "sl": validated_sl,  # Use validated stop loss
                "tp": validated_tp,  # Use validated take profit
                "deviation": int(self.max_slippage),
                "magic": self.magic_number,
                "comment": "EW_Signal_Validated",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,  # Fill or Kill - works with demo account
            }
            
            # Execute order with retries
            result = self._execute_order_with_retry(request)
            
            # Special debugging for retcode 10030
            if not result.success and result.error_code == 10030:
                self.logger.error(f"🚨 RETCODE 10030 DEBUG for {signal.symbol}:")
                self._debug_symbol_info(signal.symbol)
                self._debug_order_filling_modes(signal.symbol)
                
                # Try alternative filling modes
                result = self._retry_with_alternative_filling(request)
            
            if result.success:
                # Track position
                position = self._create_position_record(signal, position_size, result)
                if position:
                    self.active_positions[position.position_id] = position
                    self.logger.info(f"✅ Position opened: {signal.symbol} {signal.signal_type.value} "
                                   f"{position_size.lot_size} lots at {result.price:.5f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Order execution error: {e}")
            return ExecutionResult(
                success=False, order_id=None, position_id=None, price=None,
                volume=0, error_code=-1, error_message=f"Execution error: {e}",
                execution_time=datetime.now()
            )
    
    def _execute_order_with_retry(self, request: Dict) -> ExecutionResult:
        """Execute order with retry logic"""
        
        for attempt in range(self.retry_attempts):
            try:
                result = mt5.order_send(request)
                
                if result is None:
                    error = mt5.last_error()
                    self.logger.warning(f"Order attempt {attempt + 1} failed: {error}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return ExecutionResult(
                        success=False, order_id=None, position_id=None, price=None,
                        volume=request['volume'], error_code=error[0] if error else -1,
                        error_message=error[1] if error else "Unknown error",
                        execution_time=datetime.now()
                    )
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Calculate slippage
                    slippage_pips = self._calculate_slippage(request['price'], result.price, request['symbol'])
                    
                    return ExecutionResult(
                        success=True,
                        order_id=result.order,
                        position_id=result.order,  # In MT5, often same as order ID
                        price=result.price,
                        volume=result.volume,
                        error_code=result.retcode,
                        error_message="Success",
                        execution_time=datetime.now(),
                        slippage_pips=slippage_pips
                    )
                
                else:
                    error_msg = self._get_retcode_description(result.retcode)
                    self.logger.warning(f"Order attempt {attempt + 1} failed: {error_msg}")
                    
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                        continue
                    
                    return ExecutionResult(
                        success=False, order_id=None, position_id=None, price=None,
                        volume=request['volume'], error_code=result.retcode,
                        error_message=error_msg, execution_time=datetime.now()
                    )
                    
            except Exception as e:
                self.logger.error(f"Order execution exception: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                
                return ExecutionResult(
                    success=False, order_id=None, position_id=None, price=None,
                    volume=request['volume'], error_code=-1,
                    error_message=f"Exception: {e}", execution_time=datetime.now()
                )
        
        return ExecutionResult(
            success=False, order_id=None, position_id=None, price=None,
            volume=request['volume'], error_code=-1,
            error_message="All retry attempts failed", execution_time=datetime.now()
        )
    
    def close_position(self, position_id: int, reason: str = "Manual close") -> ExecutionResult:
        """Close an existing position"""
        try:
            if position_id not in self.active_positions:
                return ExecutionResult(
                    success=False, order_id=None, position_id=position_id, price=None,
                    volume=0, error_code=-1, error_message="Position not found",
                    execution_time=datetime.now()
                )
            
            position = self.active_positions[position_id]
            
            # Get current price
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return ExecutionResult(
                    success=False, order_id=None, position_id=position_id, price=None,
                    volume=0, error_code=-1, error_message="Failed to get current price",
                    execution_time=datetime.now()
                )
            
            # Determine close price and order type
            if position.type == 'buy':
                close_price = tick.bid
                order_type = mt5.ORDER_TYPE_SELL
            else:
                close_price = tick.ask
                order_type = mt5.ORDER_TYPE_BUY
            
            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position_id,
                "price": close_price,
                "deviation": int(self.max_slippage),
                "magic": self.magic_number,
                "comment": f"Close_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,  # Fill or Kill - works with demo account
            }
            
            # Execute close order
            result = self._execute_order_with_retry(request)
            
            if result.success:
                # Remove from active positions
                del self.active_positions[position_id]
                self.logger.info(f"🔒 Position closed: {position.symbol} {position.type} "
                               f"{position.volume} lots at {result.price:.5f} - {reason}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Position close error: {e}")
            return ExecutionResult(
                success=False, order_id=None, position_id=position_id, price=None,
                volume=0, error_code=-1, error_message=f"Close error: {e}",
                execution_time=datetime.now()
            )
    
    def update_position_sl_tp(self, position_id: int, new_sl: Optional[float] = None, 
                             new_tp: Optional[float] = None) -> bool:
        """Update stop loss and take profit for existing position"""
        try:
            if position_id not in self.active_positions:
                self.logger.warning(f"Position {position_id} not found for SL/TP update")
                return False
            
            position = self.active_positions[position_id]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position_id,
                "sl": new_sl if new_sl is not None else position.stop_loss,
                "tp": new_tp if new_tp is not None else position.take_profit,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Update local tracking
                if new_sl is not None:
                    position.stop_loss = new_sl
                if new_tp is not None:
                    position.take_profit = new_tp
                
                self.logger.info(f"📊 Updated SL/TP for {position.symbol}: SL={new_sl}, TP={new_tp}")
                return True
            else:
                error_msg = self._get_retcode_description(result.retcode) if result else "Unknown error"
                self.logger.error(f"Failed to update SL/TP: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"SL/TP update error: {e}")
            return False
    
    def get_position_status(self, position_id: int) -> Optional[Position]:
        """Get current position status"""
        if position_id in self.active_positions:
            position = self.active_positions[position_id]
            
            # Update current price
            tick = mt5.symbol_info_tick(position.symbol)
            if tick:
                if position.type == 'buy':
                    position.update_current_price(tick.bid)
                else:
                    position.update_current_price(tick.ask)
            
            return position
        return None
    
    def get_all_positions(self) -> Dict[int, Position]:
        """Get all active positions"""
        # Update current prices for all positions
        for position in self.active_positions.values():
            tick = mt5.symbol_info_tick(position.symbol)
            if tick:
                if position.type == 'buy':
                    position.update_current_price(tick.bid)
                else:
                    position.update_current_price(tick.ask)
        
        return self.active_positions.copy()
    
    def monitor_positions(self) -> Dict[str, any]:
        """Monitor all positions and return status summary"""
        total_positions = len(self.active_positions)
        total_profit = sum(pos.profit for pos in self.active_positions.values())
        
        # Count by type
        long_positions = sum(1 for pos in self.active_positions.values() if pos.type == 'buy')
        short_positions = total_positions - long_positions
        
        # Find positions near SL/TP
        positions_near_sl = []
        positions_near_tp = []
        
        for pos in self.active_positions.values():
            if pos.type == 'buy':
                if pos.current_price <= pos.stop_loss * 1.002:  # Within 0.2%
                    positions_near_sl.append(pos.symbol)
                if pos.current_price >= pos.take_profit * 0.998:  # Within 0.2%
                    positions_near_tp.append(pos.symbol)
            else:
                if pos.current_price >= pos.stop_loss * 0.998:
                    positions_near_sl.append(pos.symbol)
                if pos.current_price <= pos.take_profit * 1.002:
                    positions_near_tp.append(pos.symbol)
        
        return {
            'total_positions': total_positions,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'total_profit': total_profit,
            'positions_near_sl': positions_near_sl,
            'positions_near_tp': positions_near_tp,
            'last_update': datetime.now()
        }
    
    def _load_existing_positions(self):
        """Load existing MT5 positions into tracking"""
        try:
            # Clear existing tracking first
            self.active_positions.clear()
            
            positions = mt5.positions_get()
            if positions:
                loaded_count = 0
                for pos in positions:
                    if pos.magic == self.magic_number:  # Only Elliott Wave positions
                        position = Position(
                            position_id=pos.ticket,
                            symbol=pos.symbol,
                            type='buy' if pos.type == 0 else 'sell',
                            volume=pos.volume,
                            open_price=pos.price_open,
                            current_price=pos.price_current,
                            stop_loss=pos.sl,
                            take_profit=pos.tp,
                            swap=pos.swap,
                            profit=pos.profit,
                            comment=pos.comment,
                            open_time=datetime.fromtimestamp(pos.time),
                            wave_pattern="Unknown",  # Can't recover from comment
                            signal_confidence=0.0
                        )
                        self.active_positions[pos.ticket] = position
                        loaded_count += 1
                        self.logger.info(f"📍 Loaded position: {pos.symbol} {pos.volume} lots (ID: {pos.ticket})")
                
                self.logger.info(f"Loaded {loaded_count} existing Elliott Wave positions")
            else:
                self.logger.info("No existing positions found")
                
        except Exception as e:
            self.logger.error(f"Error loading existing positions: {e}")
    
    def _create_position_record(self, signal: TradingSignal, position_size: PositionSize, 
                               result: ExecutionResult) -> Optional[Position]:
        """Create position record from signal and execution result"""
        try:
            return Position(
                position_id=result.position_id,
                symbol=signal.symbol,
                type='buy' if signal.signal_type == SignalType.BUY else 'sell',
                volume=position_size.lot_size,
                open_price=result.price,
                current_price=result.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                swap=0.0,
                profit=0.0,
                comment=f"EW_{signal.wave_pattern}_{signal.current_wave}",
                open_time=result.execution_time,
                wave_pattern=signal.wave_pattern,
                signal_confidence=signal.confidence
            )
        except Exception as e:
            self.logger.error(f"Error creating position record: {e}")
            return None
    
    def _calculate_slippage(self, requested_price: float, executed_price: float, symbol: str) -> float:
        """Calculate slippage in pips"""
        if 'JPY' in symbol:
            return (executed_price - requested_price) * 100
        elif symbol in ['XAUUSD']:
            return (executed_price - requested_price) * 10
        elif symbol.startswith('US') or symbol in ['NAS100']:
            return executed_price - requested_price
        else:
            return (executed_price - requested_price) * 10000
    
    def _get_retcode_description(self, retcode: int) -> str:
        """Get human-readable description of MT5 return code"""
        retcode_descriptions = {
            mt5.TRADE_RETCODE_DONE: "Request completed",
            mt5.TRADE_RETCODE_REQUOTE: "Requote",
            mt5.TRADE_RETCODE_REJECT: "Request rejected",
            mt5.TRADE_RETCODE_CANCEL: "Request canceled",
            mt5.TRADE_RETCODE_PLACED: "Order placed",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_NO_MONEY: "Insufficient funds",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF: "Off quotes",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade disabled",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
            mt5.TRADE_RETCODE_CONNECTION: "No connection",
            mt5.TRADE_RETCODE_ONLY_REAL: "Only real accounts allowed",
            mt5.TRADE_RETCODE_LIMIT_ORDERS: "Orders limit reached",
            mt5.TRADE_RETCODE_LIMIT_VOLUME: "Volume limit reached",
            mt5.TRADE_RETCODE_INVALID_ORDER: "Invalid order",
            mt5.TRADE_RETCODE_POSITION_CLOSED: "Position closed",
            # Additional important codes
            10004: "Requote",
            10006: "Request rejected",
            10007: "Request canceled by trader",
            10008: "Order placed",
            10009: "Request completed",
            10010: "Request completed partially",
            10011: "Request processing error",
            10012: "Request canceled by timeout",
            10013: "Invalid request",
            10014: "Invalid volume in the request",
            10015: "Invalid price in the request",
            10016: "Invalid stops in the request",
            10017: "Trade is disabled",
            10018: "Market is closed",
            10019: "There is not enough money to complete the request",
            10020: "Prices changed",
            10021: "There are no quotes to process the request",
            10022: "Invalid order expiration date",
            10023: "Order state changed",
            10024: "Too frequent requests",
            10025: "No changes in request",
            10026: "Autotrading disabled by server",
            10027: "Autotrading disabled by client terminal",
            10028: "Request locked for processing",
            10029: "Order or position frozen",
            10030: "Invalid order filling type",  # THE PROBLEMATIC ONE!
            10031: "No connection with the trade server",
            10032: "Operation is allowed only for live accounts",
            10033: "The number of pending orders has reached the limit",
            10034: "The volume of orders and positions for the symbol has reached the limit",
            10035: "Incorrect or prohibited order type",
            10036: "Position with the specified POSITION_IDENTIFIER has already been closed",
        }
        
        return retcode_descriptions.get(retcode, f"Unknown retcode: {retcode}")
    
    def _debug_symbol_info(self, symbol: str):
        """Debug symbol information for retcode 10030 troubleshooting"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                self.logger.error(f"   Symbol: {symbol}")
                self.logger.error(f"   Digits: {symbol_info.digits}")
                self.logger.error(f"   Point: {symbol_info.point}")
                self.logger.error(f"   Spread: {symbol_info.spread}")
                self.logger.error(f"   Stops Level: {symbol_info.trade_stops_level}")
                self.logger.error(f"   Trade Mode: {symbol_info.trade_mode}")
                self.logger.error(f"   Filling Mode: {symbol_info.filling_mode}")
                self.logger.error(f"   Expiration Mode: {symbol_info.expiration_mode}")
                self.logger.error(f"   Min Volume: {symbol_info.volume_min}")
                self.logger.error(f"   Max Volume: {symbol_info.volume_max}")
                self.logger.error(f"   Volume Step: {symbol_info.volume_step}")
        except Exception as e:
            self.logger.error(f"   Error getting symbol info: {e}")
    
    def _debug_order_filling_modes(self, symbol: str):
        """Debug available order filling modes"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                filling_mode = symbol_info.filling_mode
                self.logger.error(f"   Available filling modes for {symbol}:")
                
                if filling_mode & mt5.SYMBOL_FILLING_FOK:
                    self.logger.error(f"   ✅ FOK (Fill or Kill)")
                if filling_mode & mt5.SYMBOL_FILLING_IOC:
                    self.logger.error(f"   ✅ IOC (Immediate or Cancel)")
                if filling_mode & mt5.SYMBOL_FILLING_RETURN:
                    self.logger.error(f"   ✅ RETURN (Return)")
                    
                if filling_mode == 0:
                    self.logger.error(f"   ❌ No filling modes available!")
                    
        except Exception as e:
            self.logger.error(f"   Error getting filling modes: {e}")
    
    def _retry_with_alternative_filling(self, original_request: Dict) -> ExecutionResult:
        """Retry order with alternative filling modes for retcode 10030"""
        symbol = original_request['symbol']
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return ExecutionResult(
                    success=False, order_id=None, position_id=None, price=None,
                    volume=original_request['volume'], error_code=10030,
                    error_message="Cannot get symbol info for alternative filling",
                    execution_time=datetime.now()
                )
            
            filling_modes = [
                (mt5.ORDER_FILLING_RETURN, "RETURN"),
                (mt5.ORDER_FILLING_IOC, "IOC"), 
                (mt5.ORDER_FILLING_FOK, "FOK")
            ]
            
            for filling_mode, mode_name in filling_modes:
                # Check if this filling mode is supported
                if not (symbol_info.filling_mode & (1 << (filling_mode - 1))):
                    self.logger.warning(f"   ⚠️ {mode_name} not supported for {symbol}")
                    continue
                
                self.logger.info(f"   🔄 Retrying {symbol} with {mode_name} filling mode...")
                
                # Create new request with different filling mode
                retry_request = original_request.copy()
                retry_request['type_filling'] = filling_mode
                
                # Execute with new filling mode
                result = self._execute_order_with_retry(retry_request)
                
                if result.success:
                    self.logger.info(f"   ✅ {symbol} successful with {mode_name} filling!")
                    return result
                else:
                    self.logger.warning(f"   ❌ {symbol} failed with {mode_name}: {result.error_message}")
            
            # All filling modes failed
            return ExecutionResult(
                success=False, order_id=None, position_id=None, price=None,
                volume=original_request['volume'], error_code=10030,
                error_message="All alternative filling modes failed",
                execution_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in alternative filling retry: {e}")
            return ExecutionResult(
                success=False, order_id=None, position_id=None, price=None,
                volume=original_request['volume'], error_code=10030,
                error_message=f"Alternative filling error: {e}",
                execution_time=datetime.now()
            )

    def _has_existing_position(self, symbol: str) -> bool:
        """Check if position already exists for this symbol"""
        try:
            # Get all current MT5 positions
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                # Check if any position has our magic number
                for pos in positions:
                    if pos.magic == self.magic_number:
                        self.logger.info(f"🔄 {symbol}: Position already exists (ID: {pos.ticket}, Volume: {pos.volume})")
                        return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking existing positions for {symbol}: {e}")
            return False  # Assume no position on error to allow trading

if __name__ == "__main__":
    # Test trade executor
    logging.basicConfig(level=logging.INFO)
    
    executor = TradeExecutor()
    if executor.connect():
        print("🔗 Trade Executor connected")
        
        # Monitor existing positions
        status = executor.monitor_positions()
        print(f"📊 Positions: {status['total_positions']} "
              f"(Long: {status['long_positions']}, Short: {status['short_positions']})")
        print(f"💰 Total P&L: {status['total_profit']:.2f}")
        
        executor.disconnect()
    else:
        print("❌ Failed to connect Trade Executor")