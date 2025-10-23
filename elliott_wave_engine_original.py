#!/usr/bin/env python3
"""
Elliott Wave Engine V2 - ORIGINAL BACKTEST LOGIC
Implementiert die exakte Elliott Wave Logik aus dem erfolgreichen Backtest
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Dir(Enum):
    UP = 1
    DOWN = -1

@dataclass
class Pivot:
    idx: int
    price: float
    kind: str  # 'H' for High, 'L' for Low

@dataclass
class Impulse:
    direction: Dir
    pivots: List[Pivot]  # 6 pivots: [p0, p1, p2, p3, p4, p5]
    confidence: float = 0.0
    
    @property
    def wave_5_end(self) -> Optional[Pivot]:
        """Return Wave 5 end pivot (should be pivot[5])"""
        return self.pivots[5] if len(self.pivots) >= 6 else None
        
    @property
    def wave_3_end(self) -> Optional[Pivot]:
        """Return Wave 3 end pivot (should be pivot[3])"""
        return self.pivots[3] if len(self.pivots) >= 4 else None
        
    @property
    def wave_4_end(self) -> Optional[Pivot]:
        """Return Wave 4 end pivot (should be pivot[4])"""
        return self.pivots[4] if len(self.pivots) >= 5 else None

@dataclass
class ABC:
    direction: Dir
    pivots: List[Pivot]  # 4 pivots
    confidence: float = 0.0
    
    @property
    def c_end(self) -> Optional[Pivot]:
        """Return C wave end pivot (should be pivot[3])"""
        return self.pivots[3] if len(self.pivots) >= 4 else None
        
    @property
    def a_end(self) -> Optional[Pivot]:
        """Return A wave end pivot (should be pivot[1])"""
        return self.pivots[1] if len(self.pivots) >= 2 else None

@dataclass 
class ElliottWavePattern:
    pattern_type: str  # 'impulse' or 'abc'
    direction: str     # 'bullish' or 'bearish'
    confidence: float
    signal_strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    pivots: List[Pivot]
    fibonacci_levels: Dict[str, float]

class ElliottWaveEngine:
    """Original Elliott Wave Engine aus dem erfolgreichen Backtest"""
    
    def __init__(self, config: Dict = None):
        """Initialize with original backtest parameters"""
        # Original aggressive profile parameters (EXACT from backtest)
        self.config = config or {
            'PRIMARY_ZZ_PCT': 0.012,      # Original aggressive
            'PRIMARY_ZZ_ATR_MULT': 0.90,  # Original aggressive
            'PRIMARY_MIN_IMP_ATR': 1.8,   # Original aggressive
            'H1_ZZ_PCT': 0.0020,          # Original aggressive
            'H1_ZZ_ATR_MULT': 0.60,       # Original aggressive
            'H1_MIN_IMP_ATR': 1.6,        # Original aggressive
            'ENTRY_ZONE_W3': (0.382, 0.786),
            'ENTRY_ZONE_W5': (0.236, 0.618),
            'ENTRY_ZONE_C': (0.382, 0.786),
            'TP1': 1.272,
            'TP2': 1.618,
            'ATR_PERIOD': 14,
            'ATR_MULT_BUFFER': 0.20       # Original aggressive
        }
        
        # Initialize engines for different timeframes
        self.primary_engine = ElliottEngine(
            self.config['PRIMARY_ZZ_PCT'],
            self.config['PRIMARY_ZZ_ATR_MULT'], 
            self.config['PRIMARY_MIN_IMP_ATR']
        )
        
        self.h1_engine = ElliottEngine(
            self.config['H1_ZZ_PCT'],
            self.config['H1_ZZ_ATR_MULT'],
            self.config['H1_MIN_IMP_ATR']
        )
        
    def analyze_elliott_waves(self, data: pd.DataFrame) -> Optional[ElliottWavePattern]:
        """
        Analyze Elliott Wave patterns using original backtest logic
        
        Args:
            data: OHLC data with columns ['open', 'high', 'low', 'close', 'date']
            
        Returns:
            ElliottWavePattern if valid pattern found, None otherwise
        """
        try:
            if data is None or len(data) < 50:
                return None
                
            # Calculate ATR
            data = self._calculate_atr(data)
            
            # Convert to numpy arrays
            close = data['close'].values
            atr = data['atr'].values
            
            # Run ZigZag analysis
            pivots = self.h1_engine.zigzag(close, atr)
            
            if len(pivots) < 6:
                logger.debug(f"Not enough pivots: {len(pivots)}")
                return None
            
            # Detect impulse patterns
            impulses = self.h1_engine.detect_impulses(pivots, close, atr)
            
            # Detect ABC patterns  
            abcs = self.h1_engine.detect_abcs(pivots)
            
            # Find best pattern
            best_pattern = self._find_best_pattern(impulses, abcs, data)
            
            if best_pattern:
                logger.info(f"Elliott Wave pattern detected: {best_pattern.pattern_type} {best_pattern.direction}")
                return best_pattern
                
            return None
            
        except Exception as e:
            logger.error(f"Elliott Wave analysis error: {e}")
            return None
    
    def analyze_waves(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Wrapper method for signal generator compatibility
        Returns dictionary with impulses, abc_corrections, and trend_direction
        """
        try:
            if data is None or len(data) < 50:
                return None
                
            # Calculate ATR
            data = self._calculate_atr(data)
            
            # Convert to numpy arrays
            close = data['close'].values
            atr = data['atr'].values
            
            # Run ZigZag analysis
            pivots = self.h1_engine.zigzag(close, atr)
            
            if len(pivots) < 6:
                logger.debug(f"Not enough pivots: {len(pivots)}")
                return None
            
            # Detect impulse patterns
            impulses = self.h1_engine.detect_impulses(pivots, close, atr)
            
            # Detect ABC patterns  
            abcs = self.h1_engine.detect_abcs(pivots)
            
            # Determine trend direction from most recent impulse
            trend_direction = None
            if impulses:
                recent_impulse = impulses[-1]  # Most recent
                trend_direction = recent_impulse.direction
            
            return {
                'impulses': impulses,
                'abc_corrections': abcs,
                'trend_direction': trend_direction,
                'pivots': pivots
            }
            
        except Exception as e:
            logger.error(f"Wave analysis error: {e}")
            return None

    def _calculate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range"""
        df = data.copy()
        
        # Calculate True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=self.config['ATR_PERIOD']).mean()
        
        return df
    
    def _find_best_pattern(self, impulses: List[Impulse], abcs: List[ABC], data: pd.DataFrame) -> Optional[ElliottWavePattern]:
        """Find the best Elliott Wave pattern for trading"""
        
        patterns = []
        
        # Evaluate impulse patterns
        for impulse in impulses[-3:]:  # Check last 3 impulses
            pattern = self._evaluate_impulse_pattern(impulse, data)
            if pattern:
                patterns.append(pattern)
        
        # Evaluate ABC patterns
        for abc in abcs[-3:]:  # Check last 3 ABCs
            pattern = self._evaluate_abc_pattern(abc, data)
            if pattern:
                patterns.append(pattern)
        
        if not patterns:
            return None
            
        # Return pattern with highest confidence
        return max(patterns, key=lambda p: p.confidence)
    
    def _evaluate_impulse_pattern(self, impulse: Impulse, data: pd.DataFrame) -> Optional[ElliottWavePattern]:
        """Evaluate impulse pattern for trading setup"""
        
        try:
            pivots = impulse.pivots
            if len(pivots) != 6:
                return None
                
            p0, p1, p2, p3, p4, p5 = pivots
            
            # Calculate wave measurements
            wave1 = abs(p1.price - p0.price)
            wave3 = abs(p3.price - p2.price)
            wave5 = abs(p5.price - p4.price)
            
            # Basic Elliott Wave rules
            if wave3 < wave1 * 0.6:  # Wave 3 should be significant
                return None
                
            # Determine direction and entry setup
            if impulse.direction == Dir.UP:
                direction = "bullish"
                
                # Check for Wave 5 completion or Wave 4 retracement entry
                current_price = data['close'].iloc[-1]
                
                # Entry in Wave 4 retracement zone
                fib_zone = self._calculate_fibonacci_zone(
                    p2.price, p3.price, impulse.direction, self.config['ENTRY_ZONE_W5']
                )
                
                if fib_zone[0] <= current_price <= fib_zone[1]:
                    entry_price = current_price
                    stop_loss = p4.price * (1 - self.config['ATR_MULT_BUFFER'])
                    take_profit = self._calculate_fibonacci_extension(
                        p0.price, p1.price, impulse.direction, self.config['TP1']
                    )
                else:
                    return None
                    
            else:  # Dir.DOWN
                direction = "bearish"
                current_price = data['close'].iloc[-1]
                
                fib_zone = self._calculate_fibonacci_zone(
                    p2.price, p3.price, impulse.direction, self.config['ENTRY_ZONE_W5']
                )
                
                if fib_zone[0] <= current_price <= fib_zone[1]:
                    entry_price = current_price
                    stop_loss = p4.price * (1 + self.config['ATR_MULT_BUFFER'])
                    take_profit = self._calculate_fibonacci_extension(
                        p0.price, p1.price, impulse.direction, self.config['TP1']
                    )
                else:
                    return None
            
            # Calculate confidence based on wave proportions
            confidence = self._calculate_impulse_confidence(impulse, data)
            
            # Calculate signal strength
            signal_strength = min(confidence * 1.2, 1.0)
            
            # Risk/Reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            if risk_reward < 1.5:  # Minimum RR requirement
                return None
            
            # Fibonacci levels for analysis
            fibonacci_levels = {
                'entry_zone_low': fib_zone[0],
                'entry_zone_high': fib_zone[1],
                'tp1': take_profit,
                'tp2': self._calculate_fibonacci_extension(p0.price, p1.price, impulse.direction, self.config['TP2'])
            }
            
            return ElliottWavePattern(
                pattern_type="impulse",
                direction=direction,
                confidence=confidence,
                signal_strength=signal_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                pivots=pivots,
                fibonacci_levels=fibonacci_levels
            )
            
        except Exception as e:
            logger.error(f"Error evaluating impulse pattern: {e}")
            return None
    
    def _evaluate_abc_pattern(self, abc: ABC, data: pd.DataFrame) -> Optional[ElliottWavePattern]:
        """Evaluate ABC correction pattern for trading setup"""
        
        try:
            pivots = abc.pivots
            if len(pivots) != 4:
                return None
                
            a0, b1, c1, end = pivots
            
            # Calculate wave measurements
            wave_a = abs(b1.price - a0.price)
            wave_b = abs(c1.price - b1.price)
            wave_c_current = abs(end.price - c1.price)
            
            # ABC retracement rules
            b_retracement = wave_b / wave_a
            if not (0.3 <= b_retracement <= 0.86):
                return None
            
            current_price = data['close'].iloc[-1]
            
            # Determine direction and entry
            if abc.direction == Dir.UP:
                direction = "bullish"
                
                # Entry in C wave completion zone
                fib_zone = self._calculate_fibonacci_zone(
                    a0.price, b1.price, abc.direction, self.config['ENTRY_ZONE_C']
                )
                
                if fib_zone[0] <= current_price <= fib_zone[1]:
                    entry_price = current_price
                    stop_loss = c1.price * (1 - self.config['ATR_MULT_BUFFER'])
                    take_profit = b1.price + (b1.price - a0.price) * 0.618
                else:
                    return None
                    
            else:  # Dir.DOWN
                direction = "bearish"
                
                fib_zone = self._calculate_fibonacci_zone(
                    a0.price, b1.price, abc.direction, self.config['ENTRY_ZONE_C']
                )
                
                if fib_zone[0] <= current_price <= fib_zone[1]:
                    entry_price = current_price
                    stop_loss = c1.price * (1 + self.config['ATR_MULT_BUFFER'])
                    take_profit = b1.price - (a0.price - b1.price) * 0.618
                else:
                    return None
            
            # Calculate confidence
            confidence = self._calculate_abc_confidence(abc, data)
            signal_strength = min(confidence * 1.1, 1.0)
            
            # Risk/Reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            if risk_reward < 1.2:  # Lower RR requirement for ABC
                return None
            
            fibonacci_levels = {
                'entry_zone_low': fib_zone[0],
                'entry_zone_high': fib_zone[1],
                'b_retracement': b_retracement,
                'tp1': take_profit
            }
            
            return ElliottWavePattern(
                pattern_type="abc",
                direction=direction,
                confidence=confidence,
                signal_strength=signal_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                pivots=pivots,
                fibonacci_levels=fibonacci_levels
            )
            
        except Exception as e:
            logger.error(f"Error evaluating ABC pattern: {e}")
            return None
    
    def _calculate_fibonacci_zone(self, price_a: float, price_b: float, direction: Dir, zone: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate Fibonacci retracement zone"""
        lo, hi = sorted(zone)
        
        if direction == Dir.UP:
            swing = price_b - price_a
            zone_low = price_b - swing * hi
            zone_high = price_b - swing * lo
        else:
            swing = price_a - price_b
            zone_low = price_b + swing * lo
            zone_high = price_b + swing * hi
            
        return (min(zone_low, zone_high), max(zone_low, zone_high))
    
    def _calculate_fibonacci_extension(self, price_a: float, price_b: float, direction: Dir, extension: float) -> float:
        """Calculate Fibonacci extension level"""
        swing = price_b - price_a
        
        if direction == Dir.UP:
            return price_b + swing * (extension - 1.0)
        else:
            return price_b - abs(swing) * (extension - 1.0)
    
    def _calculate_impulse_confidence(self, impulse: Impulse, data: pd.DataFrame) -> float:
        """Calculate confidence score for impulse pattern"""
        pivots = impulse.pivots
        
        # Wave proportion analysis
        wave1 = abs(pivots[1].price - pivots[0].price)
        wave3 = abs(pivots[3].price - pivots[2].price)
        wave5 = abs(pivots[5].price - pivots[4].price)
        
        # Ideal Elliott Wave proportions
        wave3_ratio = wave3 / wave1 if wave1 > 0 else 0
        wave5_ratio = wave5 / wave1 if wave1 > 0 else 0
        
        confidence = 0.3  # Base confidence
        
        # Wave 3 should be strongest (1.618 ideal)
        if 1.4 <= wave3_ratio <= 2.0:
            confidence += 0.3
        elif 1.0 <= wave3_ratio <= 2.5:
            confidence += 0.2
            
        # Wave 5 should be similar to Wave 1 (0.618-1.618 range)
        if 0.6 <= wave5_ratio <= 1.8:
            confidence += 0.2
            
        # Recent ATR volatility
        recent_atr = data['atr'].iloc[-10:].mean()
        if recent_atr > data['atr'].iloc[-50:].mean():
            confidence += 0.2
            
        return min(confidence, 1.0)
    
    def _calculate_abc_confidence(self, abc: ABC, data: pd.DataFrame) -> float:
        """Calculate confidence score for ABC pattern"""
        pivots = abc.pivots
        
        # Calculate retracement ratios
        wave_a = abs(pivots[1].price - pivots[0].price)
        wave_b = abs(pivots[2].price - pivots[1].price)
        
        b_ratio = wave_b / wave_a if wave_a > 0 else 0
        
        confidence = 0.3  # Base confidence
        
        # Ideal B wave retracement (0.5-0.618)
        if 0.45 <= b_ratio <= 0.65:
            confidence += 0.4
        elif 0.35 <= b_ratio <= 0.75:
            confidence += 0.2
            
        # Volume confirmation (if available)
        if 'volume' in data.columns:
            recent_volume = data['volume'].iloc[-5:].mean()
            avg_volume = data['volume'].iloc[-50:].mean()
            if recent_volume > avg_volume * 1.2:
                confidence += 0.15
                
        return min(confidence, 1.0)


class ElliottEngine:
    """Core Elliott Wave detection engine - extracted from original backtest"""
    
    def __init__(self, zz_pct: float, zz_atr_mult: float, min_impulse_atr: float):
        self.zz_pct = zz_pct
        self.zz_atr_mult = zz_atr_mult
        self.min_imp = min_impulse_atr

    @staticmethod
    def _thr(base: float, atr: float, pct: float, atr_mult: float) -> float:
        """Calculate threshold for ZigZag"""
        if pd.isna(atr):
            return base * pct
        return max(base * pct, atr * atr_mult)

    def zigzag(self, close: np.ndarray, atr: np.ndarray) -> List[Pivot]:
        """ZigZag algorithm - original implementation"""
        piv = []
        if len(close) < 3:
            return piv
            
        last = close[0]
        hi = last
        lo = last
        hi_i = 0
        lo_i = 0
        direction = None
        
        for i in range(1, len(close)):
            p = close[i]
            thr = self._thr(last, atr[i] if atr is not None and i < len(atr) else np.nan, 
                           self.zz_pct, self.zz_atr_mult)
            
            if direction in (None, Dir.UP):
                if p > hi:
                    hi = p
                    hi_i = i
                if hi - p >= thr:
                    piv.append(Pivot(hi_i, float(hi), 'H'))
                    last = hi
                    lo = p
                    lo_i = i
                    direction = Dir.DOWN
                    
            if direction in (None, Dir.DOWN):
                if p < lo:
                    lo = p
                    lo_i = i
                if p - lo >= thr:
                    piv.append(Pivot(lo_i, float(lo), 'L'))
                    last = lo
                    hi = p
                    hi_i = i
                    direction = Dir.UP
        
        # Sort and clean pivots
        piv.sort(key=lambda x: x.idx)
        cleaned = []
        for p in piv:
            if not cleaned or cleaned[-1].kind != p.kind:
                cleaned.append(p)
            else:
                if (p.kind == 'H' and p.price >= cleaned[-1].price) or \
                   (p.kind == 'L' and p.price <= cleaned[-1].price):
                    cleaned[-1] = p
                    
        return cleaned

    def detect_impulses(self, piv: List[Pivot], close: np.ndarray, atr: np.ndarray) -> List[Impulse]:
        """Detect 5-wave impulse patterns"""
        res = []
        i = 0
        
        while i <= len(piv) - 6:
            s = piv[i:i+6]
            kinds = ''.join(p.kind for p in s)
            
            if kinds == 'LHLHLH':  # Bullish impulse
                p0, p1, p2, p3, p4, p5 = s
                w1 = p1.price - p0.price
                w3 = p3.price - p2.price
                
                # Elliott Wave rules
                if p2.price <= p0.price or w1 <= 0 or w3 < 0.6 * w1:
                    i += 1
                    continue
                if p4.price <= p1.price * 0.98:
                    i += 1
                    continue
                    
                # ATR requirement
                atr_b = atr[min(p3.idx, len(atr) - 1)]
                if atr_b > 0 and (w3 / atr_b) < self.min_imp:
                    i += 1
                    continue
                    
                # Calculate basic confidence based on wave proportions
                confidence = 70.0  # Base confidence
                if w3 > w1 * 1.2:  # Strong Wave 3
                    confidence += 10
                if p5.price > p3.price * 1.01:  # Wave 5 extends properly
                    confidence += 10
                    
                res.append(Impulse(Dir.UP, [p0, p1, p2, p3, p4, p5], confidence))
                i += 3
                
            elif kinds == 'HLHLHL':  # Bearish impulse
                p0, p1, p2, p3, p4, p5 = s
                w1 = p0.price - p1.price
                w3 = p2.price - p3.price
                
                # Elliott Wave rules
                if p2.price >= p0.price or w1 <= 0 or w3 < 0.6 * w1:
                    i += 1
                    continue
                if p4.price >= p1.price * 1.02:
                    i += 1
                    continue
                    
                # ATR requirement
                atr_b = atr[min(p3.idx, len(atr) - 1)]
                if atr_b > 0 and (abs(w3) / atr_b) < self.min_imp:
                    i += 1
                    continue
                    
                # Calculate basic confidence based on wave proportions  
                confidence = 70.0  # Base confidence
                if w3 > w1 * 1.2:  # Strong Wave 3
                    confidence += 10
                if p5.price < p3.price * 0.99:  # Wave 5 extends properly down
                    confidence += 10
                    
                res.append(Impulse(Dir.DOWN, [p0, p1, p2, p3, p4, p5], confidence))
                i += 3
            else:
                i += 1
                
        return res

    def detect_abcs(self, piv: List[Pivot]) -> List[ABC]:
        """Detect ABC correction patterns"""
        out = []
        i = 0
        
        while i <= len(piv) - 4:
            s = piv[i:i+4]
            kinds = ''.join(p.kind for p in s)
            
            if kinds == 'HLHL':  # Bearish ABC
                h0, l1, h1, l2 = s
                A = h0.price - l1.price
                B = h1.price - l1.price
                
                if A <= 0 or not (0.3 <= B/A <= 0.86) or not (l2.price < l1.price):
                    i += 1
                    continue
                    
                # Calculate ABC confidence
                confidence = 65.0  # Base confidence for ABC
                ratio = B/A
                if 0.5 <= ratio <= 0.7:  # Ideal retracement
                    confidence += 15
                    
                out.append(ABC(Dir.DOWN, [h0, l1, h1, l2], confidence))
                i += 2
                
            elif kinds == 'LHLH':  # Bullish ABC
                l0, h1, l1, h2 = s
                A = h1.price - l0.price
                B = h1.price - l1.price
                
                if A <= 0 or not (0.3 <= B/A <= 0.86) or not (h2.price > h1.price):
                    i += 1
                    continue
                    
                # Calculate ABC confidence
                confidence = 65.0  # Base confidence for ABC
                ratio = B/A  
                if 0.5 <= ratio <= 0.7:  # Ideal retracement
                    confidence += 15
                    
                out.append(ABC(Dir.UP, [l0, h1, l1, h2], confidence))
                i += 2
            else:
                i += 1
                
        return out