"""
Elliott Wave Live Trading System V2
Investment Bank Architecture - Main Trading Engine
Orchestrates all modular components for professional Elliott Wave trading
"""

import json
import time
import logging
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
from dataclasses import asdict

# Import our modular components
from elliott_wave_engine_original import ElliottWaveEngine
from market_data_manager import MarketDataManager
from risk_manager import RiskManager, RiskParameters
from signal_generator import SignalGenerator, TradingSignal
from trade_executor import TradeExecutor, ExecutionResult

class ElliottWaveTradingEngine:
    """
    Main trading engine orchestrating all modules
    Professional Investment Bank architecture
    """
    
    def __init__(self, config_file: str = "elliott_live_config_v2.json", symbols_file: str = "symbols.txt"):
        self.logger = logging.getLogger(__name__)
        
        # Store file paths
        self.config_file = config_file
        self.symbols_file = symbols_file
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize modular components
        self.elliott_engine = ElliottWaveEngine()
        self.market_data = MarketDataManager()
        self.risk_manager = RiskManager(RiskParameters(**self.config.get('risk_parameters', {})))
        self.signal_generator = SignalGenerator(config=self.config)
        self.trade_executor = TradeExecutor()
        
        # Trading state
        self.is_running = False
        self.last_analysis_time = {}
        self.active_signals = {}
        self.trading_session_start = datetime.now()
        
        # Performance tracking
        self.session_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
        
        # Threading
        self.analysis_thread = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
    
    def _load_config(self, config_file: str) -> Dict:
        """Load trading configuration"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Load symbols from external file
            symbols = self._load_symbols_from_file(self.symbols_file)
            if symbols:
                config['symbols'] = symbols
                self.logger.info(f"Symbols loaded from {self.symbols_file}: {len(symbols)} symbols")
            else:
                self.logger.info(f"Using config default symbols: {len(config.get('symbols', []))} symbols")
            
            # Validate essential configuration
            required_keys = ['symbols', 'timeframes', 'scan_interval', 'account_balance']
            for key in required_keys:
                if key not in config:
                    raise KeyError(f"Missing required config key: {key}")
            
            self.logger.info(f"Configuration loaded: {len(config['symbols'])} symbols, "
                           f"{config['scan_interval']}s interval")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            # Return default configuration
            return {
                'symbols': ['EURUSD', 'XAUUSD', 'US30', 'NAS100', 'US500.f', 'AUDNOK'],
                'timeframes': [mt5.TIMEFRAME_M30],
                'scan_interval': 120,
                'account_balance': 10000,
                'risk_parameters': {
                    'max_risk_per_trade': 0.01,
                    'max_daily_risk': 0.03,
                    'max_portfolio_risk': 0.05
                },
                'trading_hours': {
                    'start': "07:00",
                    'end': "21:00",
                    'timezone': "UTC"
                }
            }
    
    def _load_symbols_from_file(self, symbols_file: str) -> List[str]:
        """Load trading symbols from external text file"""
        try:
            symbols = []
            with open(symbols_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        symbols.append(line.upper())
            
            if symbols:
                self.logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
                return symbols
            else:
                self.logger.warning(f"No symbols found in {symbols_file}")
                return []
                
        except FileNotFoundError:
            self.logger.warning(f"Symbols file {symbols_file} not found, using config defaults")
            return []
        except Exception as e:
            self.logger.error(f"Error loading symbols from {symbols_file}: {e}")
            return []
    
    def initialize(self) -> bool:
        """Initialize all trading components"""
        try:
            self.logger.info("Initializing Elliott Wave Trading Engine V2...")
            
            # Connect market data
            if not self.market_data.connect():
                self.logger.error("Failed to connect market data")
                return False
            
            # Connect trade executor
            if not self.trade_executor.connect():
                self.logger.error("Failed to connect trade executor")
                return False
            
            # Validate symbols
            valid_symbols = []
            for symbol in self.config['symbols']:
                symbol_info = self.market_data.get_symbol_info(symbol)
                if symbol_info:
                    valid_symbols.append(symbol)
                    self.logger.info(f"{symbol} validated")
                else:
                    self.logger.warning(f"{symbol} not available")
            
            if not valid_symbols:
                self.logger.error("No valid symbols available")
                return False
            
            self.config['symbols'] = valid_symbols
            
            # Initialize last analysis times
            for symbol in valid_symbols:
                self.last_analysis_time[symbol] = datetime.min
            
            self.logger.info(f"Elliott Wave Trading Engine V2 initialized successfully")
            self.logger.info(f"Symbols: {', '.join(valid_symbols)}")
            self.logger.info(f"Scan interval: {self.config['scan_interval']}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def start_trading(self):
        """Start automated trading"""
        if not self.is_running:
            self.is_running = True
            self.stop_event.clear()
            
            # Start analysis thread
            self.analysis_thread = threading.Thread(target=self._analysis_loop)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.logger.info("Elliott Wave Trading Engine V2 started")
    
    def stop_trading(self):
        """Stop automated trading"""
        if self.is_running:
            self.is_running = False
            self.stop_event.set()
            
            # Wait for threads to finish
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=10)
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
            
            self.logger.info("Elliott Wave Trading Engine V2 stopped")
    
    def _analysis_loop(self):
        """Main analysis loop - runs in separate thread with complete scan guarantee"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Check trading hours
                if not self._is_trading_hours():
                    time.sleep(60)  # Check every minute during off-hours
                    continue
                
                scan_start_time = time.time()
                symbols_processed = 0
                total_symbols = len(self.config['symbols'])
                
                self.logger.info(f"🔄 Starting scan cycle for {total_symbols} symbols...")
                
                # Analyze each symbol - COMPLETE ALL before next cycle
                for symbol in self.config['symbols']:
                    if self.stop_event.is_set():
                        break
                    
                    try:
                        symbol_start = time.time()
                        result = self._analyze_symbol(symbol)
                        symbol_time = time.time() - symbol_start
                        symbols_processed += 1
                        
                        # Log detailed result
                        if result == "skipped":
                            self.logger.info(f"⏭️ {symbol} skipped (too recent) ({symbols_processed}/{total_symbols})")
                        elif result == "no_data":
                            self.logger.info(f"❌ {symbol} no data ({symbols_processed}/{total_symbols})")
                        elif result == "analyzed":
                            self.logger.info(f"✅ {symbol} analyzed in {symbol_time:.2f}s ({symbols_processed}/{total_symbols})")
                        else:
                            self.logger.info(f"🔄 {symbol} processed in {symbol_time:.2f}s ({symbols_processed}/{total_symbols})")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error analyzing {symbol}: {e}")
                        symbols_processed += 1  # Count failed attempts too
                
                scan_duration = time.time() - scan_start_time
                
                # Count results by type
                result_counts = {}
                # This is a bit hacky but works for now - in future we could track results properly
                
                self.logger.info(f"📊 Scan completed: {symbols_processed}/{total_symbols} symbols in {scan_duration:.2f}s")
                self.logger.info(f"🚀 Speed: {symbols_processed/scan_duration:.1f} symbols/second")
                
                # Ensure minimum 120 second interval regardless of scan time
                remaining_time = self.config['scan_interval'] - scan_duration
                if remaining_time > 0:
                    self.logger.info(f"⏱️ Waiting {remaining_time:.1f}s until next scan (120s interval)")
                    time.sleep(remaining_time)
                else:
                    self.logger.warning(f"⚠️ Scan took {scan_duration:.1f}s (longer than {self.config['scan_interval']}s interval)")
                
            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
                time.sleep(10)  # Short pause on error
    
    def _analyze_symbol(self, symbol: str):
        """Analyze single symbol for Elliott Wave patterns"""
        try:
            # Check if enough time has passed since last analysis
            now = datetime.now()
            last_analysis = self.last_analysis_time.get(symbol, datetime.min)
            seconds_since_last = (now - last_analysis).total_seconds()
            
            # Skip only if analyzed in last 10 seconds (prevent spam)
            if seconds_since_last < 10:
                self.logger.debug(f"⏭️ {symbol}: Skipping - analyzed {seconds_since_last:.1f}s ago")
                return "skipped"  # Too soon since last analysis
            
            # Get market data with FORCED logging
            self.logger.info(f"📊 {symbol}: FETCHING MARKET DATA NOW...")
            df = self.market_data.get_live_data(symbol, mt5.TIMEFRAME_M30, 200)
            if df is None or not self.market_data.validate_data_quality(df, symbol):
                self.logger.warning(f"❌ {symbol}: No valid data available")
                return "no_data"
            else:
                self.logger.info(f"✅ {symbol}: GOT {len(df)} BARS OF MARKET DATA")
            
            # Get current price
            self.logger.info(f"💰 {symbol}: FETCHING CURRENT PRICE...")
            current_price = self.market_data.get_current_price(symbol)
            if current_price is None:
                self.logger.warning(f"❌ {symbol}: No current price available")
                return "no_price"
            else:
                self.logger.info(f"💰 {symbol}: CURRENT PRICE = {current_price.get('bid', 'N/A')}/{current_price.get('ask', 'N/A')}")
            
            # Generate trading signal
            signal = self.signal_generator.generate_signal(symbol, df, current_price)
            if signal is None:
                self.last_analysis_time[symbol] = now
                self.logger.debug(f"🔍 {symbol}: No signal generated")
                return "no_signal"
            
            # Risk management check
            symbol_info = self.market_data.get_symbol_info(symbol)
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                account_balance=self.config['account_balance'],
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss,
                take_profit_price=signal.take_profit,
                symbol_info=symbol_info
            )
            
            # Portfolio risk check
            can_trade, risk_reason = self.risk_manager.check_portfolio_risk(
                position_size, self.config['account_balance']
            )
            
            if not can_trade:
                self.logger.warning(f"🚫 {symbol}: Trade blocked - {risk_reason}")
                self.last_analysis_time[symbol] = now
                return "risk_blocked"
            
            # Execute trade
            if position_size.is_valid and signal.confidence >= 70:
                execution_result = self.trade_executor.execute_signal(signal, position_size)
                
                if execution_result.success:
                    # Update tracking
                    self.risk_manager.add_position(position_size)
                    self.session_stats['trades_executed'] += 1
                    
                    self.logger.info(f"🎯 {symbol}: Trade executed - {signal.signal_type.value} "
                                   f"{position_size.lot_size} lots at {execution_result.price:.5f}")
                    
                    # Log signal details
                    self._log_signal_execution(signal, position_size, execution_result)
                else:
                    self.logger.error(f"❌ {symbol}: Trade execution failed - {execution_result.error_message}")
                    return "trade_failed"
            else:
                self.logger.debug(f"🔍 {symbol}: Signal found but not executed (confidence: {signal.confidence}%)")
                return "signal_low_confidence"
            
            self.session_stats['signals_generated'] += 1
            self.last_analysis_time[symbol] = now
            return "analyzed"
            
        except Exception as e:
            self.logger.error(f"Symbol analysis error for {symbol}: {e}")
            return "error"
    
    def _monitoring_loop(self):
        """Position monitoring loop"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Monitor positions
                position_status = self.trade_executor.monitor_positions()
                
                # CRITICAL: Update active positions in signal generator
                active_symbols = set()
                if position_status and 'positions' in position_status:
                    for position in position_status['positions']:
                        active_symbols.add(position.get('symbol', ''))
                
                # Update signal generator with current active positions
                self.signal_generator.update_active_positions(active_symbols)
                
                # Update session stats
                self._update_session_stats(position_status)
                
                # Check for emergency stop conditions
                if self.risk_manager.emergency_stop():
                    self.logger.critical("🚨 EMERGENCY STOP TRIGGERED!")
                    self._emergency_close_all_positions()
                
                # Log status every 5 minutes
                if datetime.now().minute % 5 == 0:
                    self._log_status_update(position_status)
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)
    
    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        try:
            now = datetime.now()
            trading_hours = self.config.get('trading_hours', {})
            
            start_time = trading_hours.get('start', '07:00')
            end_time = trading_hours.get('end', '21:00')
            
            start_hour, start_min = map(int, start_time.split(':'))
            end_hour, end_min = map(int, end_time.split(':'))
            
            current_time = now.time()
            start = datetime.now().replace(hour=start_hour, minute=start_min, second=0).time()
            end = datetime.now().replace(hour=end_hour, minute=end_min, second=0).time()
            
            return start <= current_time <= end
            
        except Exception as e:
            self.logger.error(f"Trading hours check error: {e}")
            return True  # Default to always trading if error
    
    def _update_session_stats(self, position_status: Dict):
        """Update session statistics"""
        self.session_stats['total_pnl'] = position_status.get('total_profit', 0.0)
        
        # Calculate win rate from closed positions
        # This would require tracking completed trades
        # Simplified implementation here
        if self.session_stats['trades_executed'] > 0:
            self.session_stats['win_rate'] = (self.session_stats['successful_trades'] / 
                                            self.session_stats['trades_executed'] * 100)
    
    def _log_signal_execution(self, signal: TradingSignal, position_size, execution_result: ExecutionResult):
        """Log detailed signal execution"""
        signal_data = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal.to_dict(),
            'position_size': {
                'lot_size': position_size.lot_size,
                'risk_amount': position_size.risk_amount,
                'reward_risk_ratio': position_size.reward_risk_ratio
            },
            'execution': {
                'success': execution_result.success,
                'price': execution_result.price,
                'slippage_pips': execution_result.slippage_pips
            }
        }
        
        # Log to file for analysis
        with open(f"signals_{datetime.now().strftime('%Y%m%d')}.json", 'a') as f:
            f.write(json.dumps(signal_data) + '\n')
    
    def _log_status_update(self, position_status: Dict):
        """Log regular status updates"""
        risk_metrics = self.risk_manager.get_risk_metrics(self.config['account_balance'])
        
        self.logger.info(f"📊 Status Update:")
        self.logger.info(f"   Positions: {position_status['total_positions']} "
                        f"(Long: {position_status['long_positions']}, Short: {position_status['short_positions']})")
        self.logger.info(f"   P&L: ${position_status['total_profit']:.2f}")
        self.logger.info(f"   Portfolio Risk: {risk_metrics['total_risk_percent']:.1f}%")
        self.logger.info(f"   Signals: {self.session_stats['signals_generated']} "
                        f"| Trades: {self.session_stats['trades_executed']}")
    
    def _emergency_close_all_positions(self):
        """Emergency close all positions"""
        try:
            positions = self.trade_executor.get_all_positions()
            for position_id in positions:
                result = self.trade_executor.close_position(position_id, "Emergency stop")
                if result.success:
                    self.logger.info(f"🚨 Emergency closed position {position_id}")
                else:
                    self.logger.error(f"❌ Failed to emergency close position {position_id}")
            
            # Stop trading
            self.stop_trading()
            
        except Exception as e:
            self.logger.error(f"Emergency close error: {e}")
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        position_status = self.trade_executor.monitor_positions()
        risk_metrics = self.risk_manager.get_risk_metrics(self.config['account_balance'])
        
        session_duration = (datetime.now() - self.trading_session_start).total_seconds() / 3600
        
        return {
            'session_stats': self.session_stats.copy(),
            'position_status': position_status,
            'risk_metrics': risk_metrics,
            'session_duration_hours': session_duration,
            'average_signals_per_hour': self.session_stats['signals_generated'] / max(session_duration, 1),
            'average_trades_per_hour': self.session_stats['trades_executed'] / max(session_duration, 1),
            'is_running': self.is_running,
            'symbols_active': len(self.config['symbols']),
            'last_update': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🔄 Initiating graceful shutdown...")
        
        # Stop trading
        self.stop_trading()
        
        # Disconnect components
        self.market_data.disconnect()
        self.trade_executor.disconnect()
        
        self.logger.info("✅ Elliott Wave Trading Engine V2 shutdown complete")

def main():
    """Main entry point"""
    import sys
    import argparse
    
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Elliott Wave Live Trading System V2')
    parser.add_argument('symbols_file', nargs='?', default='symbols.txt',
                       help='Symbols file to use (default: symbols.txt)')
    parser.add_argument('--ml-threshold', '--thr', type=float, default=None,
                       help='ML threshold for signal filtering (e.g. 0.30 for top 30 percent)')
    parser.add_argument('--no-ml', action='store_true',
                       help='Disable ML filtering completely')
    parser.add_argument('--no-ema', action='store_true',
                       help='Disable EMA trend filtering')
    parser.add_argument('--config', default='elliott_live_config_v2.json',
                       help='Configuration file to use')
    parser.add_argument('--interval', type=int, default=120,
                       help='Scan interval in seconds (default: 120)')
    parser.add_argument('--max-dd-check', action='store_true',
                       help='Enable maximum drawdown monitoring')
    
    args = parser.parse_args()
    
    # Set configuration
    symbols_file = args.symbols_file
    
    print(f"🎯 Elliott Wave Trading Engine V2")
    print(f"📋 Using symbols file: {symbols_file}")
    if args.ml_threshold:
        print(f"🤖 ML Threshold: {args.ml_threshold}")
    if args.no_ml:
        print(f"🚫 ML Filtering: DISABLED")
    if args.no_ema:
        print(f"🚫 EMA Filtering: DISABLED")
    print(f"📊 Scan Interval: 120s")
    print(f"{'='*50}")
    
    # Setup logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'elliott_wave_v2_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create trading engine with custom parameters
        engine = ElliottWaveTradingEngine(
            config_file=args.config,
            symbols_file=symbols_file
        )
        
        # Apply command line overrides
        if args.ml_threshold is not None:
            engine.config['ml_threshold'] = args.ml_threshold
        if args.no_ml:
            engine.config['use_ml_filter'] = False
        if args.no_ema:
            engine.config['use_ema_filter'] = False
        if args.interval:
            engine.config['scan_interval'] = args.interval
            
        # Set max DD monitoring
        engine.config['max_dd_check'] = args.max_dd_check
        
        # Initialize
        if not engine.initialize():
            logger.error("❌ Failed to initialize trading engine")
            return
        
        # Start trading
        engine.start_trading()
        
        # Keep running until interrupted
        try:
            while engine.is_running:
                time.sleep(10)
                
                # Print status every 5 minutes
                if datetime.now().minute % 5 == 0 and datetime.now().second < 10:
                    report = engine.get_performance_report()
                    print(f"\n{'='*60}")
                    print(f"Elliott Wave Trading Engine V2 - Status Report")
                    print(f"{'='*60}")
                    print(f"Session Duration: {report['session_duration_hours']:.1f} hours")
                    print(f"Signals Generated: {report['session_stats']['signals_generated']}")
                    print(f"Trades Executed: {report['session_stats']['trades_executed']}")
                    print(f"Total P&L: ${report['session_stats']['total_pnl']:.2f}")
                    print(f"Active Positions: {report['position_status']['total_positions']}")
                    print(f"Portfolio Risk: {report['risk_metrics']['total_risk_percent']:.1f}%")
                    print(f"{'='*60}\n")
        
        except KeyboardInterrupt:
            logger.info("⚠️ Shutdown requested by user")
        
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        
        finally:
            engine.shutdown()
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()