#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultra-Optimized Backtest Runner
This script provides a wrapper for running backtests with the UltraOptimizedStrategy
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ultra_optimized_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the ultra_optimized_strategy
try:
    from ..core.ultra_optimized_strategy import UltraOptimizedStrategy
except ImportError:
    # Try to import directly
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from ultra_optimized_strategy import UltraOptimizedStrategy

class UltraOptimizedBacktest:
    """
    Wrapper class for running backtests with the UltraOptimizedStrategy
    """
    
    def __init__(self, symbol="BTCUSDT", timeframe="5m", initial_balance=50000, max_position_size=100.0):
        """
        Initialize the backtest runner
        
        Args:
            symbol (str): Symbol to backtest
            timeframe (str): Timeframe to backtest
            initial_balance (float): Initial balance
            max_position_size (float): Maximum position size
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        
        # Convert timeframe to MT5 format if needed
        self.mt5_timeframe = self._convert_timeframe(timeframe)
        
        # Initialize the strategy
        self.strategy = UltraOptimizedStrategy(
            symbol=symbol,
            timeframe=self.mt5_timeframe,
            initial_balance=initial_balance,
            max_position_size=max_position_size
        )
        
        # Create a params dictionary that can be updated by the framework
        self.strategy.params = {
            # Signal generation parameters
            'confidence_threshold': 0.85,
            'force_signals': True,
            'force_signal_threshold': 0.80,
            'min_signals_per_day': 3,
            'max_signals_per_day': 8,
            
            # Position sizing and risk management - capped for safety
            'position_size_multiplier': 3.0,
            'max_position_size': max_position_size,  # Safety cap
            'risk_per_trade': 0.01,
            
            # Trade management
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.20,
            'use_trailing_stop': True,
            'trailing_stop_activation': 0.10,
            'trailing_stop_distance': 0.05,
            
            # Advanced components - all disabled by default
            'use_markov_chains': False,
            'use_ml_models': False,
            'use_fractal_analysis': False,
            'use_microstructure': False,
            'use_volume_analysis': False,
            'use_rnn_strategy_selection': False,
            'use_bid_ask_imbalance': False,
            
            # Component weights - will be overridden
            'markov_confidence_weight': 0.0,
            'ml_confidence_weight': 0.0,
            'fractal_confidence_weight': 0.0,
            'price_action_weight': 1.0,  # Default to price action only
            'volume_confidence_weight': 0.0
        }
        
        # Set default date range
        self.strategy.start_date = None
        self.strategy.end_date = None
        
        logger.info(f"Initialized UltraOptimizedBacktest for {symbol} on {timeframe}")
        logger.info(f"Initial balance: ${initial_balance}")
        logger.info(f"Maximum position size: {max_position_size}")
    
    def _convert_timeframe(self, timeframe):
        """
        Convert timeframe string to MT5 format
        
        Args:
            timeframe (str): Timeframe string (e.g., "5m", "1h", "1d")
            
        Returns:
            str: MT5 format timeframe
        """
        # Map of common timeframes to MT5 format
        timeframe_map = {
            "1m": "M1",
            "3m": "M3",
            "5m": "M5",
            "15m": "M15",
            "30m": "M30",
            "1h": "H1",
            "4h": "H4",
            "1d": "D1",
            "1w": "W1",
            "1M": "MN1"
        }
        
        return timeframe_map.get(timeframe, timeframe)
    
    def run(self):
        """
        Run the backtest using the strategy
        
        Returns:
            dict: Backtest results
        """
        logger.info(f"Running backtest for {self.symbol} on {self.timeframe}")
        if self.strategy.start_date and self.strategy.end_date:
            logger.info(f"Date range: {self.strategy.start_date} to {self.strategy.end_date}")
        
        # Apply parameters from params dictionary to the strategy
        for param, value in self.strategy.params.items():
            if hasattr(self.strategy, param):
                setattr(self.strategy, param, value)
                logger.debug(f"Set {param} = {value}")
        
        # Run the backtest
        try:
            # Load data for the specified date range
            if self.strategy.start_date and self.strategy.end_date:
                # Load data for the specified date range
                data = self.strategy.load_data(
                    start_date=self.strategy.start_date,
                    end_date=self.strategy.end_date
                )
            else:
                # Use default date range
                data = self.strategy.load_data()
            
            if data is None or len(data) == 0:
                logger.error(f"No data available for {self.symbol} on {self.timeframe}")
                return {
                    "error": "No data available",
                    "trades": [],
                    "balance_history": [],
                    "metrics": {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "win_rate": 0,
                        "profit_factor": 0,
                        "total_profit": 0,
                        "max_drawdown": 0,
                        "avg_win": 0,
                        "avg_loss": 0,
                        "risk_reward_ratio": 0
                    }
                }
            
            # Ensure column names are consistent (convert to lowercase)
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            # Check if we need to rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns and new_name not in data.columns:
                    data.rename(columns={old_name: new_name}, inplace=True)
            
            # Run the backtest with the loaded data
            results = self.strategy.run_backtest(
                data=data,
                initial_balance=self.initial_balance,
                plot=False  # Don't plot during component testing
            )
            
            # Log summary statistics
            self._log_backtest_summary(results)
            
            return results
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "trades": [],
                "balance_history": [],
                "metrics": {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "total_profit": 0,
                    "max_drawdown": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "risk_reward_ratio": 0
                }
            }
    
    def _log_backtest_summary(self, results):
        """
        Log summary of backtest results
        
        Args:
            results (dict): Backtest results
        """
        metrics = results.get("metrics", {})
        
        logger.info(f"Backtest Summary for {self.symbol} ({self.timeframe})")
        logger.info(f"Total trades: {metrics.get('total_trades', 0)}")
        logger.info(f"Win rate: {metrics.get('win_rate', 0)*100:.2f}%")
        logger.info(f"Profit factor: {metrics.get('profit_factor', 0):.2f}")
        logger.info(f"Total profit: ${metrics.get('total_profit', 0):.2f}")
        logger.info(f"Max drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        logger.info(f"Average win: {metrics.get('avg_win', 0)*100:.2f}%")
        logger.info(f"Average loss: {metrics.get('avg_loss', 0)*100:.2f}%")
        logger.info(f"Risk-reward ratio: {metrics.get('risk_reward_ratio', 0):.2f}")
        logger.info("-" * 50)
