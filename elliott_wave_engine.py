"""
Elliott Wave Engine - Core Pattern Recognition
Ported from the original Ur-Backtest system
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class Direction(Enum):
    UP = 1
    DOWN = -1

@dataclass
class Pivot:
    idx: int
    price: float
    kind: str  # 'H' or 'L'

@dataclass 
class Impulse:
    direction: Direction
    pivots: List[Pivot]  # 6 pivots for 5-wave structure
    confidence: float = 75.0  # Default confidence
    
    def get_wave_1(self) -> Tuple[float, float]:
        return self.pivots[0].price, self.pivots[1].price
    
    def get_wave_3(self) -> Tuple[float, float]:
        return self.pivots[2].price, self.pivots[3].price
    
    def get_wave_5(self) -> Tuple[float, float]:
        return self.pivots[4].price, self.pivots[5].price
    
    # Compatibility properties for Signal Generator
    @property
    def wave_1_end(self) -> Pivot:
        return self.pivots[1] if len(self.pivots) > 1 else None
    
    @property 
    def wave_2_end(self) -> Pivot:
        return self.pivots[2] if len(self.pivots) > 2 else None
    
    @property
    def wave_3_end(self) -> Pivot:
        return self.pivots[3] if len(self.pivots) > 3 else None
    
    @property
    def wave_4_end(self) -> Pivot:
        return self.pivots[4] if len(self.pivots) > 4 else None
    
    @property
    def wave_5_end(self) -> Pivot:
        return self.pivots[5] if len(self.pivots) > 5 else None

@dataclass
class ABC:
    direction: Direction
    pivots: List[Pivot]  # 4 pivots for A-B-C structure
    confidence: float = 65.0  # Default confidence
    
    # Compatibility properties for Signal Generator
    @property
    def a_end(self) -> Pivot:
        return self.pivots[1] if len(self.pivots) > 1 else None
    
    @property
    def b_end(self) -> Pivot:
        return self.pivots[2] if len(self.pivots) > 2 else None
    
    @property
    def c_end(self) -> Pivot:
        return self.pivots[3] if len(self.pivots) > 3 else None

class ElliottWaveEngine:
    """
    Original Elliott Wave Engine from Ur-Backtest System
    Detects 5-wave impulses and ABC corrections using ZigZag
    """
    
    def __init__(self, zz_pct: float = 0.003, zz_atr_mult: float = 0.8, min_impulse_atr: float = 2.0):
        self.zz_pct = zz_pct
        self.zz_atr_mult = zz_atr_mult
        self.min_impulse_atr = min_impulse_atr
    
    def _threshold(self, base: float, atr: float, pct: float, atr_mult: float) -> float:
        """Calculate dynamic threshold for ZigZag"""
        if pd.isna(atr):
            return base * pct
        return max(base * pct, atr * atr_mult)
    
    def detect_zigzag(self, close: np.ndarray, atr: np.ndarray) -> List[Pivot]:
        """
        ZigZag pattern detection - core of Elliott Wave analysis
        """
        pivots = []
        if len(close) < 3:
            return pivots
            
        last = close[0]
        hi = last
        lo = last
        hi_i = 0
        lo_i = 0
        direction = None
        
        for i in range(1, len(close)):
            price = close[i]
            threshold = self._threshold(
                last, 
                atr[i] if atr is not None and i < len(atr) else np.nan,
                self.zz_pct, 
                self.zz_atr_mult
            )
            
            # Upward movement
            if direction in (None, Direction.UP):
                if price > hi:
                    hi = price
                    hi_i = i
                if hi - price >= threshold:
                    pivots.append(Pivot(hi_i, float(hi), 'H'))
                    last = hi
                    lo = price
                    lo_i = i
                    direction = Direction.DOWN
            
            # Downward movement  
            if direction in (None, Direction.DOWN):
                if price < lo:
                    lo = price
                    lo_i = i
                if price - lo >= threshold:
                    pivots.append(Pivot(lo_i, float(lo), 'L'))
                    last = lo
                    hi = price
                    hi_i = i
                    direction = Direction.UP
        
        # Clean up consecutive same-type pivots
        cleaned = []
        for p in pivots:
            if not cleaned or cleaned[-1].kind != p.kind:
                cleaned.append(p)
            else:
                if (p.kind == 'H' and p.price >= cleaned[-1].price) or \
                   (p.kind == 'L' and p.price <= cleaned[-1].price):
                    cleaned[-1] = p
        
        return cleaned
    
    def detect_impulses(self, pivots: List[Pivot], close: np.ndarray, atr: np.ndarray) -> List[Impulse]:
        """
        Detect 5-wave impulse patterns (1-2-3-4-5)
        """
        impulses = []
        i = 0
        
        while i <= len(pivots) - 6:
            sequence = pivots[i:i+6]
            kinds = ''.join(p.kind for p in sequence)
            
            # Bullish impulse: L-H-L-H-L-H
            if kinds == 'LHLHLH':
                p0, p1, p2, p3, p4, p5 = sequence
                
                # Wave 1 and Wave 3 validation
                wave1 = p1.price - p0.price
                wave3 = p3.price - p2.price
                
                # Basic Elliott Wave rules
                if (p2.price <= p0.price or  # Wave 2 doesn't retrace below Wave 1 start
                    wave1 <= 0 or  # Wave 1 must be positive
                    wave3 < 0.6 * wave1 or  # Wave 3 at least 60% of Wave 1
                    p4.price <= p1.price * 0.98):  # Wave 4 doesn't overlap Wave 1 (with tolerance)
                    i += 1
                    continue
                
                # ATR strength check for Wave 3
                atr_at_peak = atr[min(p3.idx, len(atr) - 1)]
                if atr_at_peak > 0 and (wave3 / atr_at_peak) < self.min_impulse_atr:
                    i += 1
                    continue
                
                impulses.append(Impulse(Direction.UP, sequence))
                i += 3  # Skip ahead to avoid overlapping patterns
                
            # Bearish impulse: H-L-H-L-H-L
            elif kinds == 'HLHLHL':
                p0, p1, p2, p3, p4, p5 = sequence
                
                # Wave 1 and Wave 3 validation
                wave1 = p0.price - p1.price
                wave3 = p2.price - p3.price
                
                # Basic Elliott Wave rules
                if (p2.price >= p0.price or  # Wave 2 doesn't retrace above Wave 1 start
                    wave1 <= 0 or  # Wave 1 must be positive
                    wave3 < 0.6 * wave1 or  # Wave 3 at least 60% of Wave 1
                    p4.price >= p1.price * 1.02):  # Wave 4 doesn't overlap Wave 1 (with tolerance)
                    i += 1
                    continue
                
                # ATR strength check for Wave 3
                atr_at_peak = atr[min(p3.idx, len(atr) - 1)]
                if atr_at_peak > 0 and (abs(wave3) / atr_at_peak) < self.min_impulse_atr:
                    i += 1
                    continue
                
                impulses.append(Impulse(Direction.DOWN, sequence))
                i += 3  # Skip ahead to avoid overlapping patterns
            else:
                i += 1
        
        return impulses
    
    def detect_abc_corrections(self, pivots: List[Pivot]) -> List[ABC]:
        """
        Detect ABC corrective patterns
        """
        corrections = []
        i = 0
        
        while i <= len(pivots) - 4:
            sequence = pivots[i:i+4]
            kinds = ''.join(p.kind for p in sequence)
            
            # Bearish ABC: H-L-H-L
            if kinds == 'HLHL':
                h0, l1, h1, l2 = sequence
                
                A = h0.price - l1.price  # A wave down
                B = h1.price - l1.price  # B wave up (retracement)
                
                # ABC validation rules
                if (A <= 0 or 
                    not (0.3 <= B/A <= 0.86) or  # B wave 30-86% retracement
                    not (l2.price < l1.price)):  # C wave extends below A
                    i += 1
                    continue
                
                corrections.append(ABC(Direction.DOWN, sequence))
                i += 2
                
            # Bullish ABC: L-H-L-H
            elif kinds == 'LHLH':
                l0, h1, l1, h2 = sequence
                
                A = h1.price - l0.price  # A wave up
                B = h1.price - l1.price  # B wave down (retracement)
                
                # ABC validation rules
                if (A <= 0 or 
                    not (0.3 <= B/A <= 0.86) or  # B wave 30-86% retracement
                    not (h2.price > h1.price)):  # C wave extends above A
                    i += 1
                    continue
                
                corrections.append(ABC(Direction.UP, sequence))
                i += 2
            else:
                i += 1
        
        return corrections
    
    @staticmethod
    def fibonacci_zone(A: float, B: float, direction: Direction, zone: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate Fibonacci retracement/extension zones"""
        lo, hi = sorted(zone)
        
        if direction == Direction.UP:
            L = B - A
            zone_low = B - L * hi
            zone_high = B - L * lo
        else:
            L = A - B
            zone_low = B + L * lo
            zone_high = B + L * hi
        
        return (min(zone_low, zone_high), max(zone_low, zone_high))
    
    @staticmethod
    def fibonacci_extension(A: float, B: float, direction: Direction, ext: float) -> float:
        """Calculate Fibonacci extension targets"""
        if direction == Direction.UP:
            return B + (B - A) * (ext - 1.0)
        else:
            return B - (A - B) * (ext - 1.0)
    
    def analyze_waves(self, df) -> dict:
        """
        Main analysis method that combines all Elliott Wave detection
        Returns comprehensive wave analysis for signal generation
        """
        try:
            if len(df) < 50:
                return {}
            
            close_values = df['close'].values
            atr_values = df['atr'].values if 'atr' in df.columns else np.full(len(df), 0.001)
            
            # Detect ZigZag pivots
            pivots = self.detect_zigzag(close_values, atr_values)
            
            if len(pivots) < 5:
                return {'pivots': pivots, 'impulses': [], 'abc_corrections': []}
            
            # Detect impulse waves
            impulses = self.detect_impulses(pivots, close_values, atr_values)
            
            # Detect ABC corrections
            abc_corrections = self.detect_abc_corrections(pivots)
            
            # Determine current trend direction
            trend_direction = self._determine_trend_direction(df)
            
            return {
                'pivots': pivots,
                'impulses': impulses,
                'abc_corrections': abc_corrections,
                'trend_direction': trend_direction,
                'last_pivot': pivots[-1] if pivots else None
            }
            
        except Exception as e:
            return {'error': str(e), 'pivots': [], 'impulses': [], 'abc_corrections': []}
    
    def _determine_trend_direction(self, df):
        """Determine overall trend direction"""
        try:
            if 'ema_fast' in df.columns and 'ema_slow' in df.columns:
                ema_fast = df['ema_fast'].iloc[-1]
                ema_slow = df['ema_slow'].iloc[-1]
                
                if ema_fast > ema_slow:
                    return Direction.UP
                else:
                    return Direction.DOWN
            else:
                # Fallback: simple price comparison
                current_price = df['close'].iloc[-1]
                old_price = df['close'].iloc[-20] if len(df) >= 20 else df['close'].iloc[0]
                
                return Direction.UP if current_price > old_price else Direction.DOWN
                
        except Exception:
            return Direction.UP  # Default

if __name__ == "__main__":
    # Test the engine with sample data
    print("Elliott Wave Engine V2 - Core Pattern Recognition")
    print("ZigZag Detection")
    print("5-Wave Impulse Recognition") 
    print("ABC Correction Detection")
    print("Fibonacci Zones & Extensions")