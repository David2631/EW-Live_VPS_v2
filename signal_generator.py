"""
Signal Generator - Elliott Wave Trading Signal Engine
Converts Elliott Wave analysis into actionable trading signals
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from enum import Enum
import logging
from datetime import datetime

from elliott_wave_engine_original import ElliottWaveEngine, Dir, Impulse, ABC

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

class SignalStrength(Enum):
    """Signal strength classification"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class TradingSignal:
    """Complete trading signal with Elliott Wave context"""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-100%
    
    # Elliott Wave context
    wave_pattern: str
    wave_direction: Dir
    current_wave: str
    fibonacci_level: Optional[float]
    
    # Technical context
    trend_alignment: bool
    momentum_confirmation: bool
    volume_confirmation: bool
    
    # Risk metrics
    stop_loss_pips: float
    take_profit_pips: float
    reward_risk_ratio: float
    
    # Metadata
    timestamp: datetime
    reasoning: str
    
    # ML metrics
    ml_score: Optional[float] = None
    ml_threshold: Optional[float] = None
    ml_passed: bool = True
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary for logging/storage"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'confidence': self.confidence,
            'wave_pattern': self.wave_pattern,
            'wave_direction': self.wave_direction.value if self.wave_direction else None,
            'current_wave': self.current_wave,
            'fibonacci_level': self.fibonacci_level,
            'reward_risk_ratio': self.reward_risk_ratio,
            'ml_score': self.ml_score,
            'ml_threshold': self.ml_threshold,
            'ml_passed': self.ml_passed,
            'timestamp': self.timestamp.isoformat(),
            'reasoning': self.reasoning
        }

class SignalGenerator:
    """
    Professional signal generator using Elliott Wave analysis
    Converts wave patterns into high-probability trading signals
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        # Initialize Elliott Wave Engine with ORIGINAL backtest settings
        self.elliott_engine = ElliottWaveEngine({
            'PRIMARY_ZZ_PCT': 0.012,      # Original (restored)
            'PRIMARY_ZZ_ATR_MULT': 0.90,  # Original (restored)
            'PRIMARY_MIN_IMP_ATR': 1.8,   # Original (restored)
            'H1_ZZ_PCT': 0.0020,          # Original (restored)
            'H1_ZZ_ATR_MULT': 0.60,       # Original (restored)
            'H1_MIN_IMP_ATR': 1.6,        # Original (restored)
            'ENTRY_ZONE_W3': (0.382, 0.786),
            'ENTRY_ZONE_W5': (0.236, 0.618),
            'ENTRY_ZONE_C': (0.382, 0.786),
            'TP1': 1.272,
            'TP2': 1.618,
            'ATR_PERIOD': 14,
            'ATR_MULT_BUFFER': 0.20       # Original (restored)
        })
        
        # Load configuration
        self.config = config or {}
        
        # Signal configuration
        self.min_confidence = 70.0  # Minimum confidence for signal generation
        self.max_spread_pips = 3.0  # Maximum spread for signal validity
        
        # Position tracking - CRITICAL: Prevent duplicate positions
        self.active_positions = set()  # Track symbols with open positions
        self.recent_signals = {}  # Track recent signals to prevent duplicates
        self.signal_cooldown = 300  # 5 minutes cooldown between same signals
        
        # ML Filter configuration
        self.use_ml_filter = self.config.get('use_ml_filter', True)
        self.ml_threshold = self.config.get('ml_threshold', 0.3)  # ORIGINAL: 30% threshold (Top 30%)
        
        # EMA Filter configuration  
        self.use_ema_filter = self.config.get('use_ema_filter', True)
        
        # Fibonacci retracement levels for entries
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    
    def update_active_positions(self, active_symbols: set):
        """Update list of symbols with active positions"""
        self.active_positions = active_symbols
    
    def _check_signal_cooldown(self, symbol: str, signal_type: str) -> bool:
        """Check if signal is in cooldown period"""
        from datetime import datetime, timedelta
        
        signal_key = f"{symbol}_{signal_type}"
        now = datetime.now()
        
        if signal_key in self.recent_signals:
            last_signal = self.recent_signals[signal_key]
            if (now - last_signal).total_seconds() < self.signal_cooldown:
                return True  # Still in cooldown
        
        # Update last signal time
        self.recent_signals[signal_key] = now
        return False  # Not in cooldown
        
    def generate_signal(self, symbol: str, df: pd.DataFrame, current_price: Dict) -> Optional[TradingSignal]:
        """
        Generate trading signal based on Elliott Wave analysis
        
        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame with technical indicators
            current_price: Current bid/ask prices
        """
        try:
            if len(df) < 100:
                self.logger.warning(f"{symbol}: Insufficient data for signal generation")
                return None
            
            # CRITICAL: Check if position already exists for this symbol
            if symbol in self.active_positions:
                self.logger.debug(f"{symbol}: Position already active - skipping signal generation")
                return None
            
            # Check spread conditions
            spread_pips = self._calculate_spread_pips(current_price, symbol)
            if spread_pips > self.max_spread_pips:
                self.logger.debug(f"{symbol}: Spread too wide ({spread_pips:.1f} pips)")
                return None
            
            # Run Elliott Wave analysis
            wave_analysis = self.elliott_engine.analyze_waves(df)
            if not wave_analysis:
                return None
            
            # Extract wave patterns
            impulses = wave_analysis.get('impulses', [])
            abc_corrections = wave_analysis.get('abc_corrections', [])
            current_trend = wave_analysis.get('trend_direction')
            
            # Generate signal based on Elliott Wave patterns
            signal = self._evaluate_elliott_patterns(
                symbol, df, current_price, impulses, abc_corrections, current_trend
            )
            
            if signal and signal.confidence >= self.min_confidence:
                # Check signal cooldown to prevent duplicates
                if self._check_signal_cooldown(symbol, signal.signal_type.value):
                    self.logger.debug(f"{symbol}: Signal {signal.signal_type.value} in cooldown period")
                    return None
                
                # Apply ML filtering if enabled
                signal = self._apply_ml_filter(signal, df)
                
                if signal and signal.ml_passed:
                    self.logger.info(f"🎯 {symbol} Signal: {signal.signal_type.value} "
                                   f"({signal.strength.value}) - Confidence: {signal.confidence:.1f}% "
                                   f"ML: {signal.ml_score:.3f}")
                    return signal
                elif signal:
                    self.logger.info(f"🚫 {symbol} Signal filtered by ML: Score {signal.ml_score:.3f} < {signal.ml_threshold:.3f}")
            
            return None
            
        except Exception as e:
            import traceback
            self.logger.error(f"Signal generation error for {symbol}: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _evaluate_elliott_patterns(self, symbol: str, df: pd.DataFrame, current_price: Dict,
                                 impulses: List[Impulse], abc_corrections: List[ABC], 
                                 trend_direction: Dir) -> Optional[TradingSignal]:
        """Evaluate Elliott Wave patterns for trading opportunities"""
        
        # DEBUG: Check actual object types and attributes
        if impulses:
            imp = impulses[0]
            self.logger.info(f"DEBUG {symbol}: Impulse type={type(imp)}, has_wave_3_end={hasattr(imp, 'wave_3_end')}, has_confidence={hasattr(imp, 'confidence')}")
        if abc_corrections:
            abc = abc_corrections[0]
            self.logger.info(f"DEBUG {symbol}: ABC type={type(abc)}, has_a_end={hasattr(abc, 'a_end')}, has_confidence={hasattr(abc, 'confidence')}")
        
        current_close = df['close'].iloc[-1]
        current_bid = current_price['bid']
        current_ask = current_price['ask']
        
        # 1. Look for Wave 5 completion (reversal signal)
        reversal_signal = self._check_wave5_completion(symbol, df, impulses, current_bid, current_ask)
        if reversal_signal:
            return reversal_signal
        
        # 2. Look for Wave 3 continuation (momentum signal)
        momentum_signal = self._check_wave3_momentum(symbol, df, impulses, current_bid, current_ask)
        if momentum_signal:
            return momentum_signal
        
        # 3. Look for ABC correction completion (trend resumption)
        correction_signal = self._check_abc_completion(symbol, df, abc_corrections, current_bid, current_ask)
        if correction_signal:
            return correction_signal
        
        # 4. Look for Wave 2/4 retracement entries
        retracement_signal = self._check_retracement_entry(symbol, df, impulses, current_bid, current_ask)
        if retracement_signal:
            return retracement_signal
        
        return None
    
    def _check_wave5_completion(self, symbol: str, df: pd.DataFrame, impulses: List[Impulse],
                               current_bid: float, current_ask: float) -> Optional[TradingSignal]:
        """Check for Wave 5 completion reversal signals"""
        
        if not impulses:
            return None
        
        # Find most recent completed 5-wave impulse
        recent_impulse = None
        for impulse in reversed(impulses):
            if impulse.wave_5_end and impulse.confidence > 75:
                recent_impulse = impulse
                break
        
        if not recent_impulse:
            return None
        
        # Check if we're near Wave 5 completion
        wave5_price = recent_impulse.wave_5_end.price
        current_close = df['close'].iloc[-1]
        
        if abs(current_close - wave5_price) / wave5_price < 0.005:  # Within 0.5%
            
            # Determine reversal direction
            if recent_impulse.direction == Dir.UP:
                # Bullish impulse completed - expect bearish reversal
                signal_type = SignalType.SELL
                entry_price = current_bid
                stop_loss = wave5_price * 1.01  # 1% above Wave 5 high
                take_profit = recent_impulse.wave_4_end.price  # Target Wave 4 level
                
            else:
                # Bearish impulse completed - expect bullish reversal
                signal_type = SignalType.BUY
                entry_price = current_ask
                stop_loss = wave5_price * 0.99  # 1% below Wave 5 low
                take_profit = recent_impulse.wave_4_end.price  # Target Wave 4 level
            
            # Calculate metrics
            # Validate stop loss and take profit meet broker requirements
            stop_loss = self._validate_stop_loss(entry_price, stop_loss, symbol)
            take_profit = self._validate_take_profit(entry_price, take_profit, symbol)
            
            stop_pips = self._calculate_pips(entry_price, stop_loss, symbol)
            tp_pips = self._calculate_pips(entry_price, take_profit, symbol)
            rr_ratio = abs(tp_pips) / abs(stop_pips) if stop_pips != 0 else 0
            
            # Signal strength based on confluence
            strength = self._calculate_signal_strength(df, recent_impulse.confidence, rr_ratio)
            confidence = min(95, recent_impulse.confidence + 10)  # Boost for Wave 5 completion
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                wave_pattern="Wave 5 Completion",
                wave_direction=recent_impulse.direction,
                current_wave="5",
                fibonacci_level=None,
                trend_alignment=True,
                momentum_confirmation=self._check_momentum_divergence(df),
                volume_confirmation=self._check_volume_confirmation(df),
                stop_loss_pips=abs(stop_pips),
                take_profit_pips=abs(tp_pips),
                reward_risk_ratio=rr_ratio,
                timestamp=datetime.now(),
                reasoning=f"Wave 5 completion reversal at {wave5_price:.5f}"
            )
        
        return None
    
    def _check_wave3_momentum(self, symbol: str, df: pd.DataFrame, impulses: List[Impulse],
                             current_bid: float, current_ask: float) -> Optional[TradingSignal]:
        """Check for Wave 3 momentum continuation signals"""
        
        if not impulses:
            return None
        
        # Look for impulse in Wave 3 development
        for impulse in reversed(impulses):
            if (impulse.wave_3_end and not impulse.wave_4_end and 
                impulse.confidence > 70):
                
                wave3_price = impulse.wave_3_end.price
                wave2_price = impulse.wave_2_end.price
                current_close = df['close'].iloc[-1]
                
                # Check if we're in early Wave 4 retracement
                if impulse.direction == Dir.UP:
                    retracement = (wave3_price - current_close) / (wave3_price - wave2_price)
                    if 0.2 < retracement < 0.5:  # 20-50% retracement
                        
                        signal_type = SignalType.BUY
                        entry_price = current_ask
                        stop_loss = wave2_price * 0.995  # Below Wave 2
                        take_profit = wave3_price * 1.618  # Wave 5 target (1.618 extension)
                        
                        # Validate stop loss and take profit meet broker requirements
                        stop_loss = self._validate_stop_loss(entry_price, stop_loss, symbol)
                        take_profit = self._validate_take_profit(entry_price, take_profit, symbol)
                        
                        stop_pips = self._calculate_pips(entry_price, stop_loss, symbol)
                        tp_pips = self._calculate_pips(entry_price, take_profit, symbol)
                        rr_ratio = abs(tp_pips) / abs(stop_pips) if stop_pips != 0 else 0
                        
                        if rr_ratio >= 2.0:  # Good risk-reward
                            strength = self._calculate_signal_strength(df, impulse.confidence, rr_ratio)
                            
                            return TradingSignal(
                                symbol=symbol,
                                signal_type=signal_type,
                                strength=strength,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                confidence=impulse.confidence,
                                wave_pattern="Wave 4 Retracement",
                                wave_direction=impulse.direction,
                                current_wave="4",
                                fibonacci_level=retracement,
                                trend_alignment=True,
                                momentum_confirmation=self._check_rsi_momentum(df),
                                volume_confirmation=self._check_volume_confirmation(df),
                                stop_loss_pips=abs(stop_pips),
                                take_profit_pips=abs(tp_pips),
                                reward_risk_ratio=rr_ratio,
                                timestamp=datetime.now(),
                                reasoning=f"Wave 4 retracement entry ({retracement:.1%} retrace)"
                            )
                
                # Similar logic for bearish Wave 3
                elif impulse.direction == Dir.DOWN:
                    retracement = (current_close - wave3_price) / (wave2_price - wave3_price)
                    if 0.2 < retracement < 0.5:
                        
                        signal_type = SignalType.SELL
                        entry_price = current_bid
                        stop_loss = wave2_price * 1.005
                        take_profit = wave3_price * 0.382  # Wave 5 target
                        
                        # Validate stop loss and take profit meet broker requirements
                        stop_loss = self._validate_stop_loss(entry_price, stop_loss, symbol)
                        take_profit = self._validate_take_profit(entry_price, take_profit, symbol)
                        
                        stop_pips = self._calculate_pips(entry_price, stop_loss, symbol)
                        tp_pips = self._calculate_pips(entry_price, take_profit, symbol)
                        rr_ratio = abs(tp_pips) / abs(stop_pips) if stop_pips != 0 else 0
                        
                        if rr_ratio >= 2.0:
                            strength = self._calculate_signal_strength(df, impulse.confidence, rr_ratio)
                            
                            return TradingSignal(
                                symbol=symbol,
                                signal_type=signal_type,
                                strength=strength,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                confidence=impulse.confidence,
                                wave_pattern="Wave 4 Retracement",
                                wave_direction=impulse.direction,
                                current_wave="4",
                                fibonacci_level=retracement,
                                trend_alignment=True,
                                momentum_confirmation=self._check_rsi_momentum(df),
                                volume_confirmation=self._check_volume_confirmation(df),
                                stop_loss_pips=abs(stop_pips),
                                take_profit_pips=abs(tp_pips),
                                reward_risk_ratio=rr_ratio,
                                timestamp=datetime.now(),
                                reasoning=f"Bearish Wave 4 retracement entry ({retracement:.1%} retrace)"
                            )
        
        return None
    
    def _check_abc_completion(self, symbol: str, df: pd.DataFrame, abc_corrections: List[ABC],
                             current_bid: float, current_ask: float) -> Optional[TradingSignal]:
        """Check for ABC correction completion signals"""
        
        if not abc_corrections:
            return None
        
        # Find recent ABC correction
        for abc in reversed(abc_corrections):
            if abc.confidence > 65 and abc.c_end:
                
                c_price = abc.c_end.price
                a_price = abc.a_end.price
                current_close = df['close'].iloc[-1]
                
                # Check if we're near C wave completion
                if abs(current_close - c_price) / c_price < 0.01:  # Within 1%
                    
                    # Determine trend resumption direction
                    if abc.direction == Dir.DOWN:  # Corrective down, expect up
                        signal_type = SignalType.BUY
                        entry_price = current_ask
                        stop_loss = c_price * 0.995
                        take_profit = a_price * 1.1  # Above A wave
                        
                    else:  # Corrective up, expect down
                        signal_type = SignalType.SELL
                        entry_price = current_bid
                        stop_loss = c_price * 1.005
                        take_profit = a_price * 0.9  # Below A wave
                    
                    # Validate stop loss and take profit meet broker requirements
                    stop_loss = self._validate_stop_loss(entry_price, stop_loss, symbol)
                    take_profit = self._validate_take_profit(entry_price, take_profit, symbol)
                    
                    stop_pips = self._calculate_pips(entry_price, stop_loss, symbol)
                    tp_pips = self._calculate_pips(entry_price, take_profit, symbol)
                    rr_ratio = abs(tp_pips) / abs(stop_pips) if stop_pips != 0 else 0
                    
                    if rr_ratio >= 1.5:
                        strength = self._calculate_signal_strength(df, abc.confidence, rr_ratio)
                        
                        return TradingSignal(
                            symbol=symbol,
                            signal_type=signal_type,
                            strength=strength,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            confidence=abc.confidence,
                            wave_pattern="ABC Correction",
                            wave_direction=abc.direction,
                            current_wave="C",
                            fibonacci_level=None,
                            trend_alignment=self._check_trend_alignment(df),
                            momentum_confirmation=self._check_rsi_momentum(df),
                            volume_confirmation=self._check_volume_confirmation(df),
                            stop_loss_pips=abs(stop_pips),
                            take_profit_pips=abs(tp_pips),
                            reward_risk_ratio=rr_ratio,
                            timestamp=datetime.now(),
                            reasoning=f"ABC correction completion at {c_price:.5f}"
                        )
        
        return None
    
    def _check_retracement_entry(self, symbol: str, df: pd.DataFrame, impulses: List[Impulse],
                                current_bid: float, current_ask: float) -> Optional[TradingSignal]:
        """Check for Fibonacci retracement entry opportunities"""
        # Implementation similar to other methods
        # Focus on 38.2%, 50%, 61.8% Fibonacci levels
        return None
    
    def _calculate_pips(self, price1: float, price2: float, symbol: str) -> float:
        """Calculate pip difference between two prices (always returns positive distance)"""
        if 'JPY' in symbol:
            return abs(price1 - price2) * 100
        elif symbol in ['XAUUSD', 'XAGUSD']:
            return abs(price1 - price2) * 10
        elif symbol.startswith('US') or symbol in ['NAS100', 'UK100', 'DE40']:
            # Indices: 1 pip = 1 point (e.g., US30: 46743.0 -> 46744.0 = 1 pip)
            return abs(price1 - price2)
        else:
            # Standard currency pairs: 0.0001 = 1 pip
            return abs(price1 - price2) * 10000
    
    def _calculate_spread_pips(self, current_price: Dict, symbol: str) -> float:
        """Calculate current spread in pips"""
        spread = current_price['spread']
        return self._calculate_pips(current_price['ask'], current_price['bid'], symbol)
    
    def _validate_stop_loss(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        """Validate and adjust stop loss to meet broker requirements"""
        
        # Calculate minimum stop distance (maximum conservative for strict brokers)
        min_stop_pips = {
            'EURUSD': 60, 'GBPUSD': 120, 'AUDUSD': 100, 'NZDUSD': 60,
            'USDCHF': 60, 'USDCAD': 80, 'USDJPY': 50,
            'XAUUSD': 200, 'XAGUSD': 120,  # Metals need larger stops
            'US30': 300, 'NAS100': 200, 'UK100': 200, 'DE40': 150,  # Indices
            'default': 80
        }
        
        # Get minimum stop for this symbol
        min_pips = min_stop_pips.get(symbol, min_stop_pips['default'])
        
        # Add spread buffer (some brokers require stop > spread + minimum)
        spread_buffer = 10  # Extra 10 pips buffer for strict brokers
        total_min_pips = min_pips + spread_buffer
        
        # Calculate current stop distance in pips  
        current_stop_pips = self._calculate_pips(entry_price, stop_loss, symbol)
        
        # If stop is too small, adjust it
        if current_stop_pips < total_min_pips:
            # Adjust stop loss to minimum distance
            if stop_loss > entry_price:  # Stop above entry (short position)
                stop_loss = entry_price + (total_min_pips / self._get_pip_factor(symbol))
            else:  # Stop below entry (long position)  
                stop_loss = entry_price - (total_min_pips / self._get_pip_factor(symbol))
                
            self.logger.info(f"{symbol}: Adjusted stop loss from {current_stop_pips:.1f} to {total_min_pips} pips (min: {min_pips} + buffer: {spread_buffer})")
        
        return stop_loss
    
    def _validate_take_profit(self, entry_price: float, take_profit: float, symbol: str) -> float:
        """Validate and adjust take profit to meet broker requirements"""
        
        # Same minimum distance requirements as stop loss (extra conservative for strict brokers)
        min_tp_pips = {
            'EURUSD': 60, 'GBPUSD': 120, 'AUDUSD': 100, 'NZDUSD': 60,
            'USDCHF': 60, 'USDCAD': 80, 'USDJPY': 50,
            'XAUUSD': 200, 'XAGUSD': 120,  # Metals need larger distances
            'US30': 300, 'NAS100': 200, 'UK100': 200, 'DE40': 150,  # Indices
            'default': 80
        }
        
        # Get minimum TP distance for this symbol
        min_pips = min_tp_pips.get(symbol, min_tp_pips['default'])
        
        # Add spread buffer
        spread_buffer = 10  # Extra 10 pips buffer for strict brokers
        total_min_pips = min_pips + spread_buffer
        
        # Calculate current TP distance in pips
        current_tp_pips = self._calculate_pips(entry_price, take_profit, symbol)
        
        # Maximum reasonable take profit distances (broker-friendly)
        max_tp_pips = {
            'EURUSD': 300, 'GBPUSD': 350, 'AUDUSD': 300, 'NZDUSD': 250,
            'USDCHF': 300, 'USDCAD': 350, 'USDJPY': 300,
            'XAUUSD': 1000, 'XAGUSD': 500,  # Metals can have larger moves
            'US30': 1500, 'NAS100': 800, 'UK100': 600, 'DE40': 500,  # Indices
            'default': 400
        }
        
        max_pips = max_tp_pips.get(symbol, max_tp_pips['default'])
        
        # Debug logging
        self.logger.info(f"{symbol}: TP Check - Current: {current_tp_pips:.1f} pips, Required: {total_min_pips} pips, Max: {max_pips} pips")
        
        # Cap take profit at maximum reasonable distance
        if current_tp_pips > max_pips:
            if take_profit > entry_price:  # TP above entry (long position)
                take_profit = entry_price + (max_pips / self._get_pip_factor(symbol))
            else:  # TP below entry (short position)  
                take_profit = entry_price - (max_pips / self._get_pip_factor(symbol))
            self.logger.info(f"{symbol}: Capped take profit from {current_tp_pips:.1f} to {max_pips} pips (max allowed)")
            current_tp_pips = max_pips
        
        # If TP is too small, adjust it
        if current_tp_pips < total_min_pips:
            # Adjust take profit to minimum distance
            if take_profit > entry_price:  # TP above entry (long position)
                take_profit = entry_price + (total_min_pips / self._get_pip_factor(symbol))
            else:  # TP below entry (short position)  
                take_profit = entry_price - (total_min_pips / self._get_pip_factor(symbol))
                
            self.logger.info(f"{symbol}: Adjusted take profit from {current_tp_pips:.1f} to {total_min_pips} pips (min: {min_pips} + buffer: {spread_buffer})")
        else:
            self.logger.info(f"{symbol}: Take profit OK at {current_tp_pips:.1f} pips")
        
        return take_profit
    
    def _get_pip_factor(self, symbol: str) -> float:
        """Get pip factor for symbol"""
        if 'JPY' in symbol:
            return 100.0  # JPY pairs: 0.01 = 1 pip
        elif symbol in ['XAUUSD', 'XAGUSD']:
            return 10.0   # Metals: 0.1 = 1 pip  
        elif symbol in ['US30', 'NAS100', 'UK100', 'DE40']:
            return 1.0    # Indices: 1.0 = 1 pip
        else:
            return 10000.0  # Standard pairs: 0.0001 = 1 pip
    
    def _calculate_signal_strength(self, df: pd.DataFrame, wave_confidence: float, rr_ratio: float) -> SignalStrength:
        """Calculate signal strength based on multiple factors"""
        score = 0
        
        # Wave confidence
        if wave_confidence > 80:
            score += 2
        elif wave_confidence > 70:
            score += 1
        
        # Risk-reward ratio
        if rr_ratio > 3:
            score += 2
        elif rr_ratio > 2:
            score += 1
        
        # Technical confluence
        if self._check_trend_alignment(df):
            score += 1
        
        if self._check_rsi_momentum(df):
            score += 1
        
        # Map score to strength
        if score >= 5:
            return SignalStrength.VERY_STRONG
        elif score >= 3:
            return SignalStrength.STRONG
        elif score >= 2:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _check_trend_alignment(self, df: pd.DataFrame) -> bool:
        """Check if price aligns with major trend"""
        # Skip EMA check if disabled
        if not self.use_ema_filter:
            return True
            
        if len(df) < 50:
            return False
        
        current_close = df['close'].iloc[-1]
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        
        return (current_close > ema_fast > ema_slow) or (current_close < ema_fast < ema_slow)
    
    def _check_rsi_momentum(self, df: pd.DataFrame) -> bool:
        """Check RSI momentum confirmation"""
        if 'rsi' not in df.columns or len(df) < 20:
            return False
        
        current_rsi = df['rsi'].iloc[-1]
        return 30 < current_rsi < 70  # Not in extreme territory
    
    def _check_momentum_divergence(self, df: pd.DataFrame) -> bool:
        """Check for momentum divergence"""
        if 'rsi' not in df.columns or len(df) < 50:
            return False
        
        # Simplified divergence check
        recent_highs = df['high'].rolling(10).max()
        recent_rsi_highs = df['rsi'].rolling(10).max()
        
        return recent_highs.iloc[-1] > recent_highs.iloc[-20] and recent_rsi_highs.iloc[-1] < recent_rsi_highs.iloc[-20]
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check volume confirmation"""
        if 'tick_volume' not in df.columns:
            return True  # Default to true if no volume data
        
        current_volume = df['tick_volume'].iloc[-1]
        avg_volume = df['tick_volume'].rolling(20).mean().iloc[-1]
        
        return current_volume > avg_volume * 0.8  # At least 80% of average volume
    
    def _apply_ml_filter(self, signal: TradingSignal, df: pd.DataFrame) -> TradingSignal:
        """Apply ML filtering to trading signal"""
        if not self.use_ml_filter:
            signal.ml_passed = True
            signal.ml_score = 1.0
            signal.ml_threshold = None
            return signal
        
        # Calculate ML score based on signal features
        ml_score = self._calculate_ml_score(signal, df)
        
        # Apply threshold
        ml_passed = ml_score >= self.ml_threshold
        
        # Update signal with ML metrics
        signal.ml_score = ml_score
        signal.ml_threshold = self.ml_threshold
        signal.ml_passed = ml_passed
        
        return signal
    
    def _calculate_ml_score(self, signal: TradingSignal, df: pd.DataFrame) -> float:
        """Calculate ML score for signal quality"""
        score = 0.0
        
        # Base score from signal confidence
        score += signal.confidence / 100.0 * 0.4
        
        # Reward/Risk ratio bonus
        if signal.reward_risk_ratio > 3.0:
            score += 0.2
        elif signal.reward_risk_ratio > 2.0:
            score += 0.1
        
        # Signal strength bonus
        if signal.strength == SignalStrength.VERY_STRONG:
            score += 0.2
        elif signal.strength == SignalStrength.STRONG:
            score += 0.1
        
        # Technical alignment bonus
        if signal.trend_alignment:
            score += 0.1
        
        if signal.momentum_confirmation:
            score += 0.1
        
        # Ensure score is between 0 and 1
        return min(max(score, 0.0), 1.0)

# Testing function

if __name__ == "__main__":
    # Test signal generator
    logging.basicConfig(level=logging.INFO)
    
    signal_gen = SignalGenerator()
    print("🎯 Signal Generator initialized")
    
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=200, freq='30T')
    sample_data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 1.1000,
        'high': np.random.randn(200).cumsum() + 1.1050,
        'low': np.random.randn(200).cumsum() + 1.0950,
        'close': np.random.randn(200).cumsum() + 1.1000,
        'tick_volume': np.random.randint(100, 1000, 200)
    }, index=dates)
    
    # Add technical indicators
    sample_data['ema_fast'] = sample_data['close'].ewm(span=50).mean()
    sample_data['ema_slow'] = sample_data['close'].ewm(span=200).mean()
    sample_data['rsi'] = 50 + np.random.randn(200) * 15
    
    current_price = {
        'bid': 1.1000,
        'ask': 1.1002,
        'spread': 0.0002
    }
    
    # Test signal generation
    signal = signal_gen.generate_signal('EURUSD', sample_data, current_price)
    if signal:
        print(f"📊 Generated signal: {signal.signal_type.value} - {signal.confidence:.1f}%")
    else:
        print("📊 No signal generated")