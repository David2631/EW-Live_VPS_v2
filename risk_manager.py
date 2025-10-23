"""
Risk Manager - Professional Position Sizing & Risk Control
Investment Bank grade risk management for Elliott Wave trading
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

@dataclass
class RiskParameters:
    """Risk configuration for trading"""
    max_risk_per_trade: float = 0.01  # 1% of account per trade
    max_daily_risk: float = 0.03      # 3% daily risk limit
    max_portfolio_risk: float = 0.05  # 5% total portfolio risk
    max_correlation_risk: float = 0.02 # 2% for correlated pairs
    max_consecutive_losses: int = 3    # Stop after 3 consecutive losses
    min_reward_risk_ratio: float = 2.0 # Minimum 2:1 RR ratio

@dataclass
class PositionSize:
    """Position sizing calculation result"""
    symbol: str
    lot_size: float
    risk_amount: float
    stop_loss_pips: float
    take_profit_pips: float
    reward_risk_ratio: float
    pip_value: float
    is_valid: bool
    reason: str = ""

class RiskManager:
    """
    Professional risk management system
    Handles position sizing, correlation, and portfolio risk
    """
    
    def __init__(self, risk_params: RiskParameters = None):
        self.logger = logging.getLogger(__name__)
        self.risk_params = risk_params or RiskParameters()
        self.daily_trades = []
        self.open_positions = {}
        self.daily_pnl = 0.0
        
        # Symbol correlation matrix (major pairs)
        self.correlation_matrix = {
            'EURUSD': {'GBPUSD': 0.85, 'AUDUSD': 0.75, 'NZDUSD': 0.70, 'USDCHF': -0.85},
            'GBPUSD': {'EURUSD': 0.85, 'AUDUSD': 0.70, 'NZDUSD': 0.65, 'USDCHF': -0.80},
            'AUDUSD': {'EURUSD': 0.75, 'GBPUSD': 0.70, 'NZDUSD': 0.85, 'USDCAD': -0.60},
            'NZDUSD': {'EURUSD': 0.70, 'GBPUSD': 0.65, 'AUDUSD': 0.85, 'USDCAD': -0.55},
            'USDCHF': {'EURUSD': -0.85, 'GBPUSD': -0.80, 'XAUUSD': -0.30},
            'USDCAD': {'AUDUSD': -0.60, 'NZDUSD': -0.55, 'XAUUSD': 0.25},
            'XAUUSD': {'USDCHF': -0.30, 'USDCAD': 0.25, 'US30': -0.20},
            'US30': {'NAS100': 0.90, 'US500.f': 0.95, 'XAUUSD': -0.20},
            'NAS100': {'US30': 0.90, 'US500.f': 0.85},
            'US500.f': {'US30': 0.95, 'NAS100': 0.85}
        }
    
    def calculate_position_size(self, symbol: str, account_balance: float, 
                               entry_price: float, stop_loss_price: float, 
                               take_profit_price: float, symbol_info: Dict) -> PositionSize:
        """
        Calculate optimal position size using Elliott Wave risk management
        
        Args:
            symbol: Trading symbol
            account_balance: Current account balance
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            symbol_info: Symbol specifications from MT5
        """
        try:
            # Calculate risk amount
            max_risk_amount = account_balance * self.risk_params.max_risk_per_trade
            
            # Calculate stop loss distance in pips
            if 'JPY' in symbol:
                pip_factor = 0.01  # JPY pairs
            elif symbol in ['XAUUSD']:
                pip_factor = 0.1   # Gold
            elif symbol.startswith('US') or symbol in ['NAS100']:
                pip_factor = 1.0   # Indices
            else:
                pip_factor = 0.0001  # Major pairs
            
            stop_loss_pips = abs(entry_price - stop_loss_price) / pip_factor
            take_profit_pips = abs(take_profit_price - entry_price) / pip_factor
            
            # Calculate pip value
            if symbol_info['currency_profit'] == 'USD':
                pip_value = symbol_info['trade_tick_value'] / symbol_info['trade_tick_size'] * pip_factor
            else:
                # Simplified calculation - would need USD conversion rate in real system
                pip_value = pip_factor * symbol_info['contract_size'] / 100000
            
            # Calculate reward-to-risk ratio
            if stop_loss_pips > 0:
                reward_risk_ratio = take_profit_pips / stop_loss_pips
            else:
                return PositionSize(symbol, 0, 0, 0, 0, 0, 0, False, "Invalid stop loss")
            
            # Validate minimum RR ratio
            if reward_risk_ratio < self.risk_params.min_reward_risk_ratio:
                return PositionSize(symbol, 0, 0, stop_loss_pips, take_profit_pips, 
                                  reward_risk_ratio, pip_value, False, 
                                  f"RR ratio {reward_risk_ratio:.2f} below minimum {self.risk_params.min_reward_risk_ratio}")
            
            # Calculate position size
            if pip_value > 0 and stop_loss_pips > 0:
                lot_size = max_risk_amount / (stop_loss_pips * pip_value)
                
                # Apply symbol constraints
                min_lot = symbol_info['volume_min']
                max_lot = symbol_info['volume_max']
                lot_step = symbol_info['volume_step']
                
                # Round to valid lot size
                lot_size = max(min_lot, min(max_lot, lot_size))
                lot_size = round(lot_size / lot_step) * lot_step
                
                # Final risk amount with actual lot size
                actual_risk = lot_size * stop_loss_pips * pip_value
                
                return PositionSize(
                    symbol=symbol,
                    lot_size=lot_size,
                    risk_amount=actual_risk,
                    stop_loss_pips=stop_loss_pips,
                    take_profit_pips=take_profit_pips,
                    reward_risk_ratio=reward_risk_ratio,
                    pip_value=pip_value,
                    is_valid=True
                )
            else:
                return PositionSize(symbol, 0, 0, 0, 0, 0, 0, False, "Invalid pip value calculation")
                
        except Exception as e:
            self.logger.error(f"Position sizing error for {symbol}: {e}")
            return PositionSize(symbol, 0, 0, 0, 0, 0, 0, False, f"Calculation error: {e}")
    
    def check_portfolio_risk(self, new_position: PositionSize, account_balance: float) -> Tuple[bool, str]:
        """Check if new position violates portfolio risk limits"""
        
        # Check daily risk limit
        total_daily_risk = self.daily_pnl + new_position.risk_amount
        if abs(total_daily_risk) > account_balance * self.risk_params.max_daily_risk:
            return False, f"Daily risk limit exceeded: {abs(total_daily_risk)/account_balance*100:.1f}%"
        
        # Check total portfolio risk
        total_portfolio_risk = sum(pos.risk_amount for pos in self.open_positions.values()) + new_position.risk_amount
        if total_portfolio_risk > account_balance * self.risk_params.max_portfolio_risk:
            return False, f"Portfolio risk limit exceeded: {total_portfolio_risk/account_balance*100:.1f}%"
        
        # Check correlation risk
        correlation_risk = self._calculate_correlation_risk(new_position)
        if correlation_risk > account_balance * self.risk_params.max_correlation_risk:
            return False, f"Correlation risk limit exceeded: {correlation_risk/account_balance*100:.1f}%"
        
        # Check consecutive losses
        if self._check_consecutive_losses():
            return False, f"Maximum consecutive losses ({self.risk_params.max_consecutive_losses}) reached"
        
        return True, "Risk checks passed"
    
    def _calculate_correlation_risk(self, new_position: PositionSize) -> float:
        """Calculate risk from correlated positions"""
        correlation_risk = 0.0
        
        for symbol, position in self.open_positions.items():
            if symbol in self.correlation_matrix.get(new_position.symbol, {}):
                correlation = abs(self.correlation_matrix[new_position.symbol][symbol])
                if correlation > 0.7:  # High correlation threshold
                    correlation_risk += position.risk_amount * correlation
        
        return correlation_risk
    
    def _check_consecutive_losses(self) -> bool:
        """Check if maximum consecutive losses reached"""
        if len(self.daily_trades) < self.risk_params.max_consecutive_losses:
            return False
        
        recent_trades = self.daily_trades[-self.risk_params.max_consecutive_losses:]
        return all(trade['pnl'] < 0 for trade in recent_trades)
    
    def update_position_status(self, symbol: str, pnl: float, is_closed: bool = False):
        """Update position status and PnL tracking"""
        if is_closed and symbol in self.open_positions:
            # Record completed trade
            trade_record = {
                'symbol': symbol,
                'pnl': pnl,
                'timestamp': datetime.now(),
                'risk_amount': self.open_positions[symbol].risk_amount
            }
            self.daily_trades.append(trade_record)
            self.daily_pnl += pnl
            
            # Remove from open positions
            del self.open_positions[symbol]
            
            self.logger.info(f"Trade closed: {symbol}, PnL: {pnl:.2f}")
    
    def add_position(self, position: PositionSize):
        """Add new position to tracking"""
        self.open_positions[position.symbol] = position
        self.logger.info(f"Position added: {position.symbol}, Lot: {position.lot_size}, Risk: {position.risk_amount:.2f}")
    
    def get_risk_metrics(self, account_balance: float) -> Dict:
        """Get current risk metrics"""
        total_risk = sum(pos.risk_amount for pos in self.open_positions.values())
        
        return {
            'open_positions': len(self.open_positions),
            'total_risk_amount': total_risk,
            'total_risk_percent': total_risk / account_balance * 100,
            'daily_pnl': self.daily_pnl,
            'daily_risk_percent': abs(self.daily_pnl) / account_balance * 100,
            'available_risk': account_balance * self.risk_params.max_portfolio_risk - total_risk,
            'daily_trades_count': len(self.daily_trades),
            'consecutive_losses': self._count_consecutive_losses()
        }
    
    def _count_consecutive_losses(self) -> int:
        """Count current consecutive losses"""
        count = 0
        for trade in reversed(self.daily_trades):
            if trade['pnl'] < 0:
                count += 1
            else:
                break
        return count
    
    def reset_daily_metrics(self):
        """Reset daily tracking (call at start of new trading day)"""
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.logger.info("Daily risk metrics reset")
    
    def emergency_stop(self) -> bool:
        """Check if emergency stop conditions are met"""
        emergency_conditions = [
            self.daily_pnl < -self.risk_params.max_daily_risk * 0.8,  # 80% of daily limit
            self._count_consecutive_losses() >= self.risk_params.max_consecutive_losses,
            len(self.open_positions) > 10  # Too many open positions
        ]
        
        return any(emergency_conditions)

if __name__ == "__main__":
    # Test risk manager
    logging.basicConfig(level=logging.INFO)
    
    risk_manager = RiskManager()
    
    # Simulate position sizing
    symbol_info = {
        'digits': 5,
        'point': 0.00001,
        'volume_min': 0.01,
        'volume_max': 100.0,
        'volume_step': 0.01,
        'trade_tick_value': 1.0,
        'trade_tick_size': 0.00001,
        'contract_size': 100000,
        'currency_profit': 'USD'
    }
    
    position = risk_manager.calculate_position_size(
        symbol='EURUSD',
        account_balance=10000,
        entry_price=1.1000,
        stop_loss_price=1.0950,  # 50 pips SL
        take_profit_price=1.1100, # 100 pips TP
        symbol_info=symbol_info
    )
    
    print(f"📊 Position Size: {position.lot_size:.2f} lots")
    print(f"💰 Risk Amount: ${position.risk_amount:.2f}")
    print(f"📈 R:R Ratio: {position.reward_risk_ratio:.2f}")
    print(f"✅ Valid: {position.is_valid}")
    
    if position.is_valid:
        # Check portfolio risk
        can_trade, reason = risk_manager.check_portfolio_risk(position, 10000)
        print(f"🔒 Risk Check: {can_trade} - {reason}")