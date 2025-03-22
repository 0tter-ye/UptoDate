#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Component Backtesting Framework
This script provides a comprehensive backtesting framework for the ultra-optimized strategy
with advanced quantitative components, supporting multi-period testing, component-specific
analysis, Monte Carlo simulations, stress testing, and cross-validation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import multiprocessing
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the backtest runner and advanced components
from .run_ultra_optimized_backtest import run_comprehensive_backtest, generate_optimal_parameters
# Check if advanced_quant_components exists in the backtest directory, otherwise import from core
try:
    from .advanced_quant_components import create_advanced_components
except ImportError:
    # Try to import from core directory
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from advanced_quant_components import create_advanced_components

# Import the ultra_optimized_strategy from core directory
try:
    from ..core.ultra_optimized_strategy import UltraOptimizedStrategy
except ImportError:
    # Try to import directly
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from ultra_optimized_strategy import UltraOptimizedStrategy

class AdvancedBacktestFramework:
    """
    Advanced backtesting framework for ultra-optimized strategy with advanced quantitative components
    """
    
    def __init__(self, symbols=None, timeframes=None, initial_balance=50000):
        """
        Initialize the advanced backtest framework
        
        Args:
            symbols (list): List of symbols to backtest
            timeframes (list): List of timeframes to backtest
            initial_balance (float): Initial balance
        """
        # Default symbols and timeframes if not provided
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        self.timeframes = timeframes or ["3m", "5m", "15m"]
        self.initial_balance = initial_balance
        
        # Set up test periods
        self.current_date = datetime.now()
        self.test_periods = {
            "1_month": {
                "start_date": (self.current_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                "end_date": self.current_date.strftime('%Y-%m-%d'),
                "description": "Recent market conditions (1 month)"
            },
            "6_months": {
                "start_date": (self.current_date - timedelta(days=180)).strftime('%Y-%m-%d'),
                "end_date": self.current_date.strftime('%Y-%m-%d'),
                "description": "Medium-term performance (6 months)"
            },
            "1_year": {
                "start_date": (self.current_date - timedelta(days=365)).strftime('%Y-%m-%d'),
                "end_date": self.current_date.strftime('%Y-%m-%d'),
                "description": "Full market cycle (1 year)"
            },
            "5_years": {
                "start_date": (self.current_date - timedelta(days=1825)).strftime('%Y-%m-%d'),
                "end_date": self.current_date.strftime('%Y-%m-%d'),
                "description": "Long-term robustness (5 years)"
            }
        }
        
        # Results storage
        self.results = {}
        self.component_results = {}
        self.monte_carlo_results = {}
        self.stress_test_results = {}
        self.cross_validation_results = {}
        
        # Component weights for testing
        self.component_weights = {
            "default": {
                'markov_confidence_weight': 0.30,
                'ml_confidence_weight': 0.25,
                'fractal_confidence_weight': 0.20,
                'price_action_weight': 0.15,
                'volume_confidence_weight': 0.10
            },
            "markov_focused": {
                'markov_confidence_weight': 0.50,
                'ml_confidence_weight': 0.20,
                'fractal_confidence_weight': 0.15,
                'price_action_weight': 0.10,
                'volume_confidence_weight': 0.05
            },
            "ml_focused": {
                'markov_confidence_weight': 0.20,
                'ml_confidence_weight': 0.50,
                'fractal_confidence_weight': 0.15,
                'price_action_weight': 0.10,
                'volume_confidence_weight': 0.05
            },
            "fractal_focused": {
                'markov_confidence_weight': 0.20,
                'ml_confidence_weight': 0.15,
                'fractal_confidence_weight': 0.50,
                'price_action_weight': 0.10,
                'volume_confidence_weight': 0.05
            },
            "balanced": {
                'markov_confidence_weight': 0.20,
                'ml_confidence_weight': 0.20,
                'fractal_confidence_weight': 0.20,
                'price_action_weight': 0.20,
                'volume_confidence_weight': 0.20
            }
        }
        
        logger.info(f"Initialized Advanced Backtest Framework")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Timeframes: {self.timeframes}")
        logger.info(f"Test periods: {len(self.test_periods)} periods from 1 month to 5 years")
    
    def run_multi_period_backtest(self, optimization_rounds=3):
        """
        Run backtests across multiple time periods
        
        Args:
            optimization_rounds (int): Number of optimization rounds to perform
            
        Returns:
            dict: Results for each period
        """
        logger.info("Starting multi-period backtesting")
        
        period_results = {}
        
        for period_name, period_info in self.test_periods.items():
            logger.info(f"Running backtest for period: {period_info['description']}")
            logger.info(f"Date range: {period_info['start_date']} to {period_info['end_date']}")
            
            # Run comprehensive backtest for this period
            results, best_config = run_comprehensive_backtest(
                symbols=self.symbols,
                timeframes=self.timeframes,
                initial_balance=self.initial_balance,
                start_date=period_info['start_date'],
                end_date=period_info['end_date'],
                optimization_rounds=optimization_rounds
            )
            
            # Store results
            period_results[period_name] = {
                "results": results,
                "best_config": best_config,
                "period_info": period_info
            }
            
            # Log summary
            self._log_period_summary(period_name, results)
        
        self.results["multi_period"] = period_results
        return period_results
    
    def _log_period_summary(self, period_name, results):
        """
        Log summary of results for a period
        
        Args:
            period_name (str): Name of the period
            results (dict): Results for the period
        """
        logger.info(f"Summary for period: {period_name}")
        
        # Calculate aggregate metrics
        total_profit = 0
        total_trades = 0
        winning_trades = 0
        total_symbols = 0
        
        for symbol, symbol_results in results.items():
            total_symbols += 1
            for timeframe, timeframe_results in symbol_results.items():
                total_profit += timeframe_results.get('net_profit', 0)
                total_trades += timeframe_results.get('total_trades', 0)
                winning_trades += timeframe_results.get('winning_trades', 0)
        
        # Log aggregate metrics
        logger.info(f"Total symbols tested: {total_symbols}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Total profit: ${total_profit:.2f}")
        logger.info(f"Overall win rate: {(winning_trades / total_trades * 100) if total_trades > 0 else 0:.2f}%")
        logger.info("-" * 50)

    def run_component_specific_tests(self, period_name="1_year", optimization_rounds=2):
        """
        Run tests for each advanced component individually to measure their impact
        
        Args:
            period_name (str): Which period to use for testing
            optimization_rounds (int): Number of optimization rounds
            
        Returns:
            dict: Results for each component
        """
        logger.info("Starting component-specific testing")
        
        if period_name not in self.test_periods:
            logger.error(f"Invalid period name: {period_name}")
            return None
        
        period_info = self.test_periods[period_name]
        logger.info(f"Using period: {period_info['description']}")
        logger.info(f"Date range: {period_info['start_date']} to {period_info['end_date']}")
        
        # Components to test individually
        components = [
            {
                "name": "baseline",
                "description": "Baseline strategy without advanced components",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False
                }
            },
            {
                "name": "markov_chains",
                "description": "Markov chain models only",
                "config": {
                    'use_markov_chains': True,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False,
                    'markov_confidence_weight': 1.0
                }
            },
            {
                "name": "ml_models",
                "description": "ML assessment and rescoring only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': True,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False,
                    'ml_confidence_weight': 1.0
                }
            },
            {
                "name": "fractal_analysis",
                "description": "Fractal dimensions and Hurst exponent only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': True,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False,
                    'fractal_confidence_weight': 1.0
                }
            },
            {
                "name": "microstructure",
                "description": "Microstructure noise analysis only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': True,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False
                }
            },
            {
                "name": "volume_analysis",
                "description": "Volume analysis only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': True,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False,
                    'volume_confidence_weight': 1.0
                }
            },
            {
                "name": "rnn_strategy",
                "description": "RNN strategy selection via pseudo-forest only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': True,
                    'use_bid_ask_imbalance': False
                }
            },
            {
                "name": "bid_ask_imbalance",
                "description": "Bid/ask volume imbalance with L2/L3 data only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': True
                }
            },
            {
                "name": "full_ensemble",
                "description": "All components with default weights",
                "config": {
                    'use_markov_chains': True,
                    'use_ml_models': True,
                    'use_fractal_analysis': True,
                    'use_microstructure': True,
                    'use_volume_analysis': True,
                    'use_rnn_strategy_selection': True,
                    'use_bid_ask_imbalance': True,
                    'markov_confidence_weight': 0.30,
                    'ml_confidence_weight': 0.25,
                    'fractal_confidence_weight': 0.20,
                    'price_action_weight': 0.15,
                    'volume_confidence_weight': 0.10
                }
            }
        ]
        
        component_results = {}
        
        # Test each component configuration
        for component in components:
            logger.info(f"Testing component: {component['name']} - {component['description']}")
            
            # Create a custom parameter set with this component configuration
            custom_params = self._get_base_parameters()
            custom_params.update(component['config'])
            
            # Ensure position size is capped at 100.0 for safety
            custom_params['max_position_size'] = 100.0
            
            # Run backtest with this component configuration
            results = self._run_component_backtest(
                component['name'],
                custom_params,
                period_info['start_date'],
                period_info['end_date'],
                optimization_rounds
            )
            
            # Store results
            component_results[component['name']] = {
                "description": component['description'],
                "config": component['config'],
                "results": results
            }
            
            # Log component summary
            self._log_component_summary(component['name'], component['description'], results)
        
        # Calculate performance improvement for each component compared to baseline
        if 'baseline' in component_results and 'full_ensemble' in component_results:
            logger.info("Calculating component performance contributions")
            
            baseline_results = component_results['baseline']['results']
            full_results = component_results['full_ensemble']['results']
            
            for component_name, component_data in component_results.items():
                if component_name not in ['baseline', 'full_ensemble']:
                    component_results[component_name]['contribution'] = self._calculate_contribution(
                        baseline_results,
                        component_data['results'],
                        full_results
                    )
        
        self.component_results = component_results
        return component_results
    
    def _get_base_parameters(self):
        """
        Get base parameters for testing
        
        Returns:
            dict: Base parameters
        """
        return {
            # Signal generation parameters
            'confidence_threshold': 0.85,
            'force_signals': True,
            'force_signal_threshold': 0.80,
            'min_signals_per_day': 3,
            'max_signals_per_day': 8,
            
            # Position sizing and risk management - capped for safety
            'position_size_multiplier': 3.0,
            'max_position_size': 100.0,  # Safety cap
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
    
    def _run_component_backtest(self, component_name, params, start_date, end_date, optimization_rounds):
        """
        Run backtest for a specific component configuration
        
        Args:
            component_name (str): Name of the component
            params (dict): Parameters for the backtest
            start_date (str): Start date
            end_date (str): End date
            optimization_rounds (int): Number of optimization rounds
            
        Returns:
            dict: Backtest results
        """
        logger.info(f"Running backtest for component: {component_name}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Create a temporary directory for results
        os.makedirs(f"component_tests/{component_name}", exist_ok=True)
        
        # Aggregate results across symbols and timeframes
        aggregate_results = {}
        
        for symbol in self.symbols:
            aggregate_results[symbol] = {}
            
            for timeframe in self.timeframes:
                logger.info(f"Testing {component_name} on {symbol} {timeframe}")
                
                try:
                    # Initialize backtest runner with custom parameters
                    from .ultra_optimized_backtest import UltraOptimizedBacktest
                    backtest = UltraOptimizedBacktest(
                        symbol=symbol, 
                        timeframe=timeframe, 
                        initial_balance=self.initial_balance
                    )
                    
                    # Set date range
                    backtest.strategy.start_date = start_date
                    backtest.strategy.end_date = end_date
                    
                    # Apply component-specific parameters
                    backtest.strategy.params.update(params)
                    
                    # Run backtest
                    results = backtest.run()
                    
                    # Store results
                    aggregate_results[symbol][timeframe] = results
                    
                except Exception as e:
                    logger.error(f"Error testing {component_name} on {symbol} {timeframe}: {str(e)}")
                    aggregate_results[symbol][timeframe] = {"error": str(e)}
        
        return aggregate_results
    
    def _log_component_summary(self, component_name, description, results):
        """
        Log summary of results for a component
        
        Args:
            component_name (str): Name of the component
            description (str): Description of the component
            results (dict): Results for the component
        """
        logger.info(f"Summary for component: {component_name} - {description}")
        
        # Calculate aggregate metrics
        total_profit = 0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        profit_factor = 0
        max_drawdown = 0
        
        for symbol, symbol_results in results.items():
            for timeframe, timeframe_results in symbol_results.items():
                if isinstance(timeframe_results, dict) and "error" not in timeframe_results:
                    total_profit += timeframe_results.get('net_profit', 0)
                    total_trades += timeframe_results.get('total_trades', 0)
                    winning_trades += timeframe_results.get('winning_trades', 0)
                    
                    # Track maximum drawdown
                    if 'max_drawdown_pct' in timeframe_results:
                        max_drawdown = max(max_drawdown, timeframe_results['max_drawdown_pct'])
                    
                    # Calculate profit factor
                    if 'gross_profit' in timeframe_results and 'gross_loss' in timeframe_results:
                        if timeframe_results['gross_loss'] != 0:
                            symbol_pf = timeframe_results['gross_profit'] / abs(timeframe_results['gross_loss'])
                            profit_factor = max(profit_factor, symbol_pf)
        
        # Log aggregate metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Total profit: ${total_profit:.2f}")
        logger.info(f"Win rate: {win_rate:.2f}%")
        logger.info(f"Profit factor: {profit_factor:.2f}")
        logger.info(f"Max drawdown: {max_drawdown*100:.2f}%")
        logger.info("-" * 50)
    
    def _calculate_contribution(self, baseline_results, component_results, full_results):
        """
        Calculate the performance contribution of a component
        
        Args:
            baseline_results (dict): Baseline results
            component_results (dict): Component results
            full_results (dict): Full ensemble results
            
        Returns:
            dict: Contribution metrics
        """
        # Extract key metrics
        baseline_profit = self._extract_total_profit(baseline_results)
        component_profit = self._extract_total_profit(component_results)
        full_profit = self._extract_total_profit(full_results)
        
        baseline_win_rate = self._extract_win_rate(baseline_results)
        component_win_rate = self._extract_win_rate(component_results)
        full_win_rate = self._extract_win_rate(full_results)
        
        # Calculate absolute improvement
        profit_improvement = component_profit - baseline_profit
        win_rate_improvement = component_win_rate - baseline_win_rate
        
        # Calculate relative contribution to full ensemble
        if full_profit != baseline_profit:
            profit_contribution = profit_improvement / (full_profit - baseline_profit) * 100
        else:
            profit_contribution = 0
            
        if full_win_rate != baseline_win_rate:
            win_rate_contribution = win_rate_improvement / (full_win_rate - baseline_win_rate) * 100
        else:
            win_rate_contribution = 0
        
        return {
            "profit_improvement": profit_improvement,
            "win_rate_improvement": win_rate_improvement,
            "profit_contribution_pct": profit_contribution,
            "win_rate_contribution_pct": win_rate_contribution
        }
    
    def _extract_total_profit(self, results):
        """
        Extract total profit from results
        
        Args:
            results (dict): Results dictionary
            
        Returns:
            float: Total profit
        """
        total_profit = 0
        
        for symbol, symbol_results in results.items():
            for timeframe, timeframe_results in symbol_results.items():
                if isinstance(timeframe_results, dict) and "error" not in timeframe_results:
                    total_profit += timeframe_results.get('net_profit', 0)
        
        return total_profit
    
    def _extract_win_rate(self, results):
        """
        Extract win rate from results
        
        Args:
            results (dict): Results dictionary
            
        Returns:
            float: Win rate (0-1)
        """
        total_trades = 0
        winning_trades = 0
        
        for symbol, symbol_results in results.items():
            for timeframe, timeframe_results in symbol_results.items():
                if isinstance(timeframe_results, dict) and "error" not in timeframe_results:
                    total_trades += timeframe_results.get('total_trades', 0)
                    winning_trades += timeframe_results.get('winning_trades', 0)
        
        return winning_trades / total_trades if total_trades > 0 else 0

    def run_monte_carlo_simulations(self, period_name="1_year", num_simulations=1000, component_name="full_ensemble"):
        """
        Run Monte Carlo simulations to analyze the distribution of outcomes
        
        Args:
            period_name (str): Which period to use for testing
            num_simulations (int): Number of Monte Carlo simulations to run
            component_name (str): Which component configuration to use
            
        Returns:
            dict: Monte Carlo simulation results
        """
        logger.info(f"Starting Monte Carlo simulations ({num_simulations} runs)")
        
        if period_name not in self.test_periods:
            logger.error(f"Invalid period name: {period_name}")
            return None
        
        period_info = self.test_periods[period_name]
        logger.info(f"Using period: {period_info['description']}")
        logger.info(f"Date range: {period_info['start_date']} to {period_info['end_date']}")
        
        # Get base parameters for the selected component
        if not hasattr(self, 'component_results') or not self.component_results:
            logger.info("No component results found, running component-specific tests first")
            self.run_component_specific_tests(period_name)
        
        if component_name not in self.component_results:
            logger.error(f"Component {component_name} not found in results")
            return None
        
        base_config = self.component_results[component_name]['config']
        base_params = self._get_base_parameters()
        base_params.update(base_config)
        
        # Ensure position size is capped at 100.0 for safety
        base_params['max_position_size'] = 100.0
        
        # Parameters to vary in Monte Carlo simulations
        param_ranges = {
            'confidence_threshold': (0.75, 0.95, 0.01),  # (min, max, step)
            'force_signal_threshold': (0.70, 0.90, 0.01),
            'position_size_multiplier': (1.0, 5.0, 0.25),
            'risk_per_trade': (0.005, 0.02, 0.001),
            'stop_loss_pct': (0.03, 0.08, 0.005),
            'take_profit_pct': (0.15, 0.35, 0.01),
            'trailing_stop_activation': (0.05, 0.20, 0.01),
            'trailing_stop_distance': (0.03, 0.10, 0.005)
        }
        
        # Component-specific parameter ranges
        if 'use_markov_chains' in base_config and base_config['use_markov_chains']:
            param_ranges['markov_confidence_weight'] = (0.1, 0.5, 0.05)
            
        if 'use_ml_models' in base_config and base_config['use_ml_models']:
            param_ranges['ml_confidence_weight'] = (0.1, 0.5, 0.05)
            param_ranges['ml_prediction_threshold'] = (0.7, 0.9, 0.05)
            
        if 'use_fractal_analysis' in base_config and base_config['use_fractal_analysis']:
            param_ranges['fractal_confidence_weight'] = (0.1, 0.5, 0.05)
        
        # Generate random parameter sets
        param_sets = []
        for i in range(num_simulations):
            param_set = base_params.copy()
            
            # Randomize parameters
            for param, (min_val, max_val, step) in param_ranges.items():
                steps = int((max_val - min_val) / step) + 1
                random_value = min_val + random.randint(0, steps - 1) * step
                param_set[param] = round(random_value, 6)  # Round to avoid floating point issues
            
            # Normalize component weights if applicable
            self._normalize_component_weights(param_set)
            
            # Ensure position size is capped at 100.0 for safety
            param_set['max_position_size'] = 100.0
            
            param_sets.append(param_set)
        
        # Run simulations
        simulation_results = []
        
        # Use a subset of symbols and timeframes for faster Monte Carlo
        mc_symbols = self.symbols[:2] if len(self.symbols) > 2 else self.symbols
        mc_timeframes = self.timeframes[:2] if len(self.timeframes) > 2 else self.timeframes
        
        logger.info(f"Running Monte Carlo with {len(mc_symbols)} symbols and {len(mc_timeframes)} timeframes")
        
        for i, param_set in enumerate(param_sets):
            if i % 100 == 0:
                logger.info(f"Running simulation {i+1}/{num_simulations}")
            
            # Run backtest with this parameter set
            results = self._run_monte_carlo_simulation(
                i,
                param_set,
                mc_symbols,
                mc_timeframes,
                period_info['start_date'],
                period_info['end_date']
            )
            
            # Store results
            simulation_results.append({
                "id": i,
                "params": param_set,
                "results": results,
                "metrics": self._calculate_simulation_metrics(results)
            })
        
        # Analyze simulation results
        analysis = self._analyze_monte_carlo_results(simulation_results)
        
        # Store results
        monte_carlo_results = {
            "component": component_name,
            "period": period_name,
            "num_simulations": num_simulations,
            "simulation_results": simulation_results,
            "analysis": analysis
        }
        
        self.monte_carlo_results[f"{component_name}_{period_name}"] = monte_carlo_results
        
        # Log summary
        self._log_monte_carlo_summary(monte_carlo_results)
        
        return monte_carlo_results
    
    def _normalize_component_weights(self, param_set):
        """
        Normalize component weights to sum to 1.0
        
        Args:
            param_set (dict): Parameter set to normalize
        """
        weight_params = [
            'markov_confidence_weight',
            'ml_confidence_weight',
            'fractal_confidence_weight',
            'price_action_weight',
            'volume_confidence_weight'
        ]
        
        # Check if weights exist in param_set
        weights = [param_set.get(p, 0) for p in weight_params if p in param_set]
        
        if weights:
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                for param in weight_params:
                    if param in param_set:
                        param_set[param] = param_set[param] / total_weight
    
    def _run_monte_carlo_simulation(self, sim_id, params, symbols, timeframes, start_date, end_date):
        """
        Run a single Monte Carlo simulation
        
        Args:
            sim_id (int): Simulation ID
            params (dict): Parameters for the simulation
            symbols (list): List of symbols to test
            timeframes (list): List of timeframes to test
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            dict: Simulation results
        """
        logger.info(f"Running Monte Carlo simulation {sim_id}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Aggregate results across symbols and timeframes
        aggregate_results = {}
        
        for symbol in symbols:
            aggregate_results[symbol] = {}
            
            for timeframe in timeframes:
                try:
                    # Initialize backtest runner with custom parameters
                    from ultra_optimized_strategy import UltraOptimizedBacktest
                    backtest = UltraOptimizedBacktest(
                        symbol=symbol, 
                        timeframe=timeframe, 
                        initial_balance=self.initial_balance
                    )
                    
                    # Set date range
                    backtest.strategy.start_date = start_date
                    backtest.strategy.end_date = end_date
                    
                    # Apply simulation parameters
                    backtest.strategy.params.update(params)
                    
                    # Run backtest
                    results = backtest.run()
                    
                    # Store results
                    aggregate_results[symbol][timeframe] = results
                    
                except Exception as e:
                    logger.error(f"Error in simulation {sim_id} on {symbol} {timeframe}: {str(e)}")
                    aggregate_results[symbol][timeframe] = {"error": str(e)}
        
        return aggregate_results
    
    def _calculate_simulation_metrics(self, results):
        """
        Calculate metrics for a simulation
        
        Args:
            results (dict): Simulation results
            
        Returns:
            dict: Metrics
        """
        total_profit = 0
        total_trades = 0
        winning_trades = 0
        max_drawdown = 0
        profit_factor = 0
        gross_profit = 0
        gross_loss = 0
        
        for symbol, symbol_results in results.items():
            for timeframe, timeframe_results in symbol_results.items():
                if isinstance(timeframe_results, dict) and "error" not in timeframe_results:
                    total_profit += timeframe_results.get('net_profit', 0)
                    total_trades += timeframe_results.get('total_trades', 0)
                    winning_trades += timeframe_results.get('winning_trades', 0)
                    
                    # Track maximum drawdown
                    if 'max_drawdown_pct' in timeframe_results:
                        max_drawdown = max(max_drawdown, timeframe_results['max_drawdown_pct'])
                    
                    # Track profit and loss
                    gross_profit += timeframe_results.get('gross_profit', 0)
                    gross_loss += abs(timeframe_results.get('gross_loss', 0))
        
        # Calculate metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "total_profit": total_profit,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss
        }
    
    def _analyze_monte_carlo_results(self, simulation_results):
        """
        Analyze Monte Carlo simulation results
        
        Args:
            simulation_results (list): List of simulation results
            
        Returns:
            dict: Analysis results
        """
        # Extract metrics
        profits = [r['metrics']['total_profit'] for r in simulation_results]
        win_rates = [r['metrics']['win_rate'] for r in simulation_results]
        drawdowns = [r['metrics']['max_drawdown'] for r in simulation_results]
        profit_factors = [r['metrics']['profit_factor'] for r in simulation_results]
        
        # Calculate percentiles
        profit_percentiles = {
            "min": min(profits),
            "p5": np.percentile(profits, 5),
            "p25": np.percentile(profits, 25),
            "p50": np.percentile(profits, 50),
            "p75": np.percentile(profits, 75),
            "p95": np.percentile(profits, 95),
            "max": max(profits)
        }
        
        win_rate_percentiles = {
            "min": min(win_rates),
            "p5": np.percentile(win_rates, 5),
            "p25": np.percentile(win_rates, 25),
            "p50": np.percentile(win_rates, 50),
            "p75": np.percentile(win_rates, 75),
            "p95": np.percentile(win_rates, 95),
            "max": max(win_rates)
        }
        
        drawdown_percentiles = {
            "min": min(drawdowns),
            "p5": np.percentile(drawdowns, 5),
            "p25": np.percentile(drawdowns, 25),
            "p50": np.percentile(drawdowns, 50),
            "p75": np.percentile(drawdowns, 75),
            "p95": np.percentile(drawdowns, 95),
            "max": max(drawdowns)
        }
        
        # Find best performing parameter sets
        best_profit_sim = max(simulation_results, key=lambda r: r['metrics']['total_profit'])
        best_risk_adjusted_sim = max(simulation_results, key=lambda r: r['metrics']['profit_factor'])
        best_win_rate_sim = max(simulation_results, key=lambda r: r['metrics']['win_rate'])
        
        # Parameter sensitivity analysis
        param_sensitivity = self._analyze_parameter_sensitivity(simulation_results)
        
        return {
            "profit_percentiles": profit_percentiles,
            "win_rate_percentiles": win_rate_percentiles,
            "drawdown_percentiles": drawdown_percentiles,
            "best_profit_sim": {
                "id": best_profit_sim['id'],
                "profit": best_profit_sim['metrics']['total_profit'],
                "params": best_profit_sim['params']
            },
            "best_risk_adjusted_sim": {
                "id": best_risk_adjusted_sim['id'],
                "profit_factor": best_risk_adjusted_sim['metrics']['profit_factor'],
                "params": best_risk_adjusted_sim['params']
            },
            "best_win_rate_sim": {
                "id": best_win_rate_sim['id'],
                "win_rate": best_win_rate_sim['metrics']['win_rate'],
                "params": best_win_rate_sim['params']
            },
            "param_sensitivity": param_sensitivity
        }
    
    def _analyze_parameter_sensitivity(self, simulation_results):
        """
        Analyze parameter sensitivity
        
        Args:
            simulation_results (list): List of simulation results
            
        Returns:
            dict: Parameter sensitivity analysis
        """
        # Get list of all parameters
        all_params = set()
        for sim in simulation_results:
            all_params.update(sim['params'].keys())
        
        # Calculate correlation between each parameter and profit
        sensitivity = {}
        
        for param in all_params:
            # Skip parameters that don't vary
            values = [sim['params'].get(param, 0) for sim in simulation_results]
            if len(set(values)) <= 1:
                continue
            
            # Calculate correlation with profit
            profits = [sim['metrics']['total_profit'] for sim in simulation_results]
            correlation = np.corrcoef(values, profits)[0, 1]
            
            # Calculate correlation with win rate
            win_rates = [sim['metrics']['win_rate'] for sim in simulation_results]
            win_rate_corr = np.corrcoef(values, win_rates)[0, 1]
            
            # Calculate correlation with drawdown
            drawdowns = [sim['metrics']['max_drawdown'] for sim in simulation_results]
            drawdown_corr = np.corrcoef(values, drawdowns)[0, 1]
            
            sensitivity[param] = {
                "profit_correlation": correlation,
                "win_rate_correlation": win_rate_corr,
                "drawdown_correlation": drawdown_corr
            }
        
        # Sort by absolute correlation with profit
        sorted_sensitivity = dict(sorted(
            sensitivity.items(),
            key=lambda item: abs(item[1]['profit_correlation']),
            reverse=True
        ))
        
        return sorted_sensitivity
    
    def _log_monte_carlo_summary(self, monte_carlo_results):
        """
        Log summary of Monte Carlo simulation results
        
        Args:
            monte_carlo_results (dict): Monte Carlo results
        """
        analysis = monte_carlo_results['analysis']
        
        logger.info(f"Monte Carlo Simulation Summary ({monte_carlo_results['num_simulations']} runs)")
        logger.info(f"Component: {monte_carlo_results['component']}")
        logger.info(f"Period: {monte_carlo_results['period']}")
        
        logger.info("\nProfit Distribution:")
        logger.info(f"Min: ${analysis['profit_percentiles']['min']:.2f}")
        logger.info(f"5th percentile: ${analysis['profit_percentiles']['p5']:.2f}")
        logger.info(f"Median: ${analysis['profit_percentiles']['p50']:.2f}")
        logger.info(f"95th percentile: ${analysis['profit_percentiles']['p95']:.2f}")
        logger.info(f"Max: ${analysis['profit_percentiles']['max']:.2f}")
        
        logger.info("\nWin Rate Distribution:")
        logger.info(f"Min: {analysis['win_rate_percentiles']['min']*100:.2f}%")
        logger.info(f"5th percentile: {analysis['win_rate_percentiles']['p5']*100:.2f}%")
        logger.info(f"Median: {analysis['win_rate_percentiles']['p50']*100:.2f}%")
        logger.info(f"95th percentile: {analysis['win_rate_percentiles']['p95']*100:.2f}%")
        logger.info(f"Max: {analysis['win_rate_percentiles']['max']*100:.2f}%")
        
        logger.info("\nDrawdown Distribution:")
        logger.info(f"Min: {analysis['drawdown_percentiles']['min']*100:.2f}%")
        logger.info(f"5th percentile: {analysis['drawdown_percentiles']['p5']*100:.2f}%")
        logger.info(f"Median: {analysis['drawdown_percentiles']['p50']*100:.2f}%")
        logger.info(f"95th percentile: {analysis['drawdown_percentiles']['p95']*100:.2f}%")
        logger.info(f"Max: {analysis['drawdown_percentiles']['max']*100:.2f}%")
        
        logger.info("\nBest Profit Parameters:")
        for param, value in analysis['best_profit_sim']['params'].items():
            logger.info(f"{param}: {value}")
        
        logger.info("\nParameter Sensitivity (correlation with profit):")
        for param, data in list(analysis['param_sensitivity'].items())[:10]:  # Top 10
            logger.info(f"{param}: {data['profit_correlation']:.4f}")
        
        logger.info("-" * 50)

    def run_stress_tests(self, period_name="1_year", component_name="full_ensemble"):
        """
        Run stress tests to evaluate strategy performance under extreme market conditions
        
        Args:
            period_name (str): Which period to use for testing
            component_name (str): Which component configuration to use
            
        Returns:
            dict: Stress test results
        """
        logger.info("Starting stress testing")
        
        if period_name not in self.test_periods:
            logger.error(f"Invalid period name: {period_name}")
            return None
        
        period_info = self.test_periods[period_name]
        logger.info(f"Using period: {period_info['description']}")
        logger.info(f"Date range: {period_info['start_date']} to {period_info['end_date']}")
        
        # Get base parameters for the selected component
        if not hasattr(self, 'component_results') or not self.component_results:
            logger.info("No component results found, running component-specific tests first")
            self.run_component_specific_tests(period_name)
        
        if component_name not in self.component_results:
            logger.error(f"Component {component_name} not found in results")
            return None
        
        base_config = self.component_results[component_name]['config']
        base_params = self._get_base_parameters()
        base_params.update(base_config)
        
        # Ensure position size is capped at 100.0 for safety
        base_params['max_position_size'] = 100.0
        
        # Define stress test scenarios
        stress_scenarios = [
            {
                "name": "baseline",
                "description": "Baseline scenario without stress",
                "modifiers": {}
            },
            {
                "name": "flash_crash",
                "description": "Flash crash scenario (sudden 20% drop)",
                "modifiers": {
                    "price_shock": -0.20,
                    "price_shock_duration": 5,  # minutes
                    "volatility_multiplier": 3.0
                }
            },
            {
                "name": "high_volatility",
                "description": "Extreme volatility scenario (3x normal)",
                "modifiers": {
                    "volatility_multiplier": 3.0,
                    "spread_multiplier": 2.0
                }
            },
            {
                "name": "low_liquidity",
                "description": "Low liquidity scenario (wide spreads, slippage)",
                "modifiers": {
                    "spread_multiplier": 5.0,
                    "slippage_multiplier": 3.0,
                    "execution_delay": 2  # seconds
                }
            },
            {
                "name": "connectivity_issues",
                "description": "Intermittent connectivity issues",
                "modifiers": {
                    "execution_delay": 5,  # seconds
                    "order_failure_rate": 0.20,  # 20% of orders fail
                    "data_gaps": True
                }
            },
            {
                "name": "rapid_reversal",
                "description": "Rapid market reversal (V-shaped recovery)",
                "modifiers": {
                    "price_shock": -0.15,
                    "price_shock_duration": 10,  # minutes
                    "reversal_magnitude": 0.25,
                    "reversal_duration": 30  # minutes
                }
            },
            {
                "name": "extreme_ranging",
                "description": "Extreme ranging market (high noise, low signal)",
                "modifiers": {
                    "noise_multiplier": 3.0,
                    "trend_strength_multiplier": 0.3
                }
            },
            {
                "name": "combined_stress",
                "description": "Combined stress scenario (multiple adverse conditions)",
                "modifiers": {
                    "volatility_multiplier": 2.5,
                    "spread_multiplier": 3.0,
                    "slippage_multiplier": 2.0,
                    "execution_delay": 3,  # seconds
                    "order_failure_rate": 0.10,  # 10% of orders fail
                    "data_gaps": True
                }
            }
        ]
        
        stress_test_results = {}
        
        # Run each stress test scenario
        for scenario in stress_scenarios:
            logger.info(f"Running stress test: {scenario['name']} - {scenario['description']}")
            
            # Create scenario-specific parameters
            scenario_params = base_params.copy()
            scenario_params.update({
                "stress_test_scenario": scenario['name'],
                "stress_test_modifiers": scenario['modifiers']
            })
            
            # Ensure position size is capped at 100.0 for safety
            scenario_params['max_position_size'] = 100.0
            
            # Run backtest with this scenario
            results = self._run_stress_test(
                scenario['name'],
                scenario_params,
                period_info['start_date'],
                period_info['end_date']
            )
            
            # Store results
            stress_test_results[scenario['name']] = {
                "description": scenario['description'],
                "modifiers": scenario['modifiers'],
                "results": results,
                "metrics": self._calculate_stress_test_metrics(results)
            }
            
            # Log scenario summary
            self._log_stress_test_summary(scenario['name'], scenario['description'], results)
        
        # Analyze stress test results
        analysis = self._analyze_stress_test_results(stress_test_results)
        
        # Store results
        stress_results = {
            "component": component_name,
            "period": period_name,
            "scenarios": stress_test_results,
            "analysis": analysis
        }
        
        self.stress_test_results[f"{component_name}_{period_name}"] = stress_results
        
        return stress_results
    
    def _run_stress_test(self, scenario_name, params, start_date, end_date):
        """
        Run a stress test scenario
        
        Args:
            scenario_name (str): Name of the scenario
            params (dict): Parameters for the scenario
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            dict: Stress test results
        """
        logger.info(f"Running stress test scenario: {scenario_name}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Create a temporary directory for results
        os.makedirs(f"stress_tests/{scenario_name}", exist_ok=True)
        
        # Aggregate results across symbols and timeframes
        aggregate_results = {}
        
        for symbol in self.symbols:
            aggregate_results[symbol] = {}
            
            for timeframe in self.timeframes:
                logger.info(f"Testing {scenario_name} on {symbol} {timeframe}")
                
                try:
                    # Initialize backtest runner with stress test parameters
                    from ultra_optimized_strategy import UltraOptimizedBacktest
                    backtest = UltraOptimizedBacktest(
                        symbol=symbol, 
                        timeframe=timeframe, 
                        initial_balance=self.initial_balance
                    )
                    
                    # Set date range
                    backtest.strategy.start_date = start_date
                    backtest.strategy.end_date = end_date
                    
                    # Apply stress test parameters
                    backtest.strategy.params.update(params)
                    
                    # Enable stress testing mode
                    backtest.strategy.enable_stress_testing(
                        scenario_name,
                        params.get('stress_test_modifiers', {})
                    )
                    
                    # Run backtest
                    results = backtest.run()
                    
                    # Store results
                    aggregate_results[symbol][timeframe] = results
                    
                except Exception as e:
                    logger.error(f"Error in stress test {scenario_name} on {symbol} {timeframe}: {str(e)}")
                    aggregate_results[symbol][timeframe] = {"error": str(e)}
        
        return aggregate_results
    
    def _calculate_stress_test_metrics(self, results):
        """
        Calculate metrics for a stress test
        
        Args:
            results (dict): Stress test results
            
        Returns:
            dict: Metrics
        """
        # Similar to simulation metrics but with additional stress-specific metrics
        metrics = self._calculate_simulation_metrics(results)
        
        # Additional stress-specific metrics
        max_consecutive_losses = 0
        recovery_time = 0
        worst_drawdown_duration = 0
        
        for symbol, symbol_results in results.items():
            for timeframe, timeframe_results in symbol_results.items():
                if isinstance(timeframe_results, dict) and "error" not in timeframe_results:
                    # Max consecutive losses
                    if 'max_consecutive_losses' in timeframe_results:
                        max_consecutive_losses = max(max_consecutive_losses, timeframe_results['max_consecutive_losses'])
                    
                    # Recovery time (in bars)
                    if 'recovery_time' in timeframe_results:
                        recovery_time = max(recovery_time, timeframe_results['recovery_time'])
                    
                    # Worst drawdown duration (in bars)
                    if 'worst_drawdown_duration' in timeframe_results:
                        worst_drawdown_duration = max(worst_drawdown_duration, timeframe_results['worst_drawdown_duration'])
        
        # Add stress-specific metrics
        metrics.update({
            "max_consecutive_losses": max_consecutive_losses,
            "recovery_time": recovery_time,
            "worst_drawdown_duration": worst_drawdown_duration
        })
        
        return metrics
    
    def _log_stress_test_summary(self, scenario_name, description, results):
        """
        Log summary of stress test results
        
        Args:
            scenario_name (str): Name of the scenario
            description (str): Description of the scenario
            results (dict): Results for the scenario
        """
        logger.info(f"Summary for stress test: {scenario_name} - {description}")
        
        # Calculate aggregate metrics
        total_profit = 0
        total_trades = 0
        winning_trades = 0
        max_drawdown = 0
        
        for symbol, symbol_results in results.items():
            for timeframe, timeframe_results in symbol_results.items():
                if isinstance(timeframe_results, dict) and "error" not in timeframe_results:
                    total_profit += timeframe_results.get('net_profit', 0)
                    total_trades += timeframe_results.get('total_trades', 0)
                    winning_trades += timeframe_results.get('winning_trades', 0)
                    
                    # Track maximum drawdown
                    if 'max_drawdown_pct' in timeframe_results:
                        max_drawdown = max(max_drawdown, timeframe_results['max_drawdown_pct'])
        
        # Log aggregate metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Total profit: ${total_profit:.2f}")
        logger.info(f"Win rate: {win_rate:.2f}%")
        logger.info(f"Max drawdown: {max_drawdown*100:.2f}%")
        logger.info("-" * 50)
    
    def _analyze_stress_test_results(self, stress_test_results):
        """
        Analyze stress test results
        
        Args:
            stress_test_results (dict): Stress test results
            
        Returns:
            dict: Analysis results
        """
        # Get baseline results
        if "baseline" not in stress_test_results:
            logger.error("Baseline scenario not found in stress test results")
            return {}
        
        baseline_metrics = stress_test_results["baseline"]["metrics"]
        
        # Calculate performance degradation for each scenario
        performance_impact = {}
        
        for scenario_name, scenario_data in stress_test_results.items():
            if scenario_name == "baseline":
                continue
            
            scenario_metrics = scenario_data["metrics"]
            
            # Calculate performance impact
            profit_change_pct = ((scenario_metrics["total_profit"] - baseline_metrics["total_profit"]) / 
                               abs(baseline_metrics["total_profit"])) * 100 if baseline_metrics["total_profit"] != 0 else 0
            
            win_rate_change = scenario_metrics["win_rate"] - baseline_metrics["win_rate"]
            drawdown_change = scenario_metrics["max_drawdown"] - baseline_metrics["max_drawdown"]
            
            performance_impact[scenario_name] = {
                "profit_change_pct": profit_change_pct,
                "win_rate_change": win_rate_change,
                "drawdown_change": drawdown_change,
                "description": scenario_data["description"]
            }
        
        # Sort scenarios by profit impact (worst first)
        sorted_impact = dict(sorted(
            performance_impact.items(),
            key=lambda item: item[1]['profit_change_pct']
        ))
        
        # Calculate robustness score (higher is better)
        # Based on how well the strategy maintains performance under stress
        robustness_scores = {}
        
        for scenario_name, impact in sorted_impact.items():
            # Profit resilience (0-100, higher is better)
            profit_resilience = max(0, 100 + impact["profit_change_pct"])
            
            # Win rate resilience (0-100, higher is better)
            win_rate_resilience = max(0, 100 - abs(impact["win_rate_change"] * 100))
            
            # Drawdown resilience (0-100, higher is better)
            drawdown_resilience = max(0, 100 - abs(impact["drawdown_change"] * 100))
            
            # Overall robustness score (weighted average)
            robustness_score = (
                profit_resilience * 0.5 +
                win_rate_resilience * 0.3 +
                drawdown_resilience * 0.2
            )
            
            robustness_scores[scenario_name] = {
                "profit_resilience": profit_resilience,
                "win_rate_resilience": win_rate_resilience,
                "drawdown_resilience": drawdown_resilience,
                "overall_score": robustness_score
            }
        
        # Calculate overall robustness score
        overall_robustness = sum(score["overall_score"] for score in robustness_scores.values()) / len(robustness_scores)
        
        # Identify most vulnerable scenarios
        vulnerable_scenarios = list(sorted_impact.items())[:3]  # Top 3 worst performing
        
        return {
            "performance_impact": sorted_impact,
            "robustness_scores": robustness_scores,
            "overall_robustness": overall_robustness,
            "vulnerable_scenarios": vulnerable_scenarios
        }
    
    def run_cross_validation(self, period_name="1_year", component_name="full_ensemble", folds=5):
        """
        Run cross-validation to prevent overfitting
        
        Args:
            period_name (str): Which period to use for testing
            component_name (str): Which component configuration to use
            folds (int): Number of folds for cross-validation
            
        Returns:
            dict: Cross-validation results
        """
        logger.info(f"Starting cross-validation with {folds} folds")
        
        if period_name not in self.test_periods:
            logger.error(f"Invalid period name: {period_name}")
            return None
        
        period_info = self.test_periods[period_name]
        logger.info(f"Using period: {period_info['description']}")
        logger.info(f"Date range: {period_info['start_date']} to {period_info['end_date']}")
        
        # Get base parameters for the selected component
        if not hasattr(self, 'component_results') or not self.component_results:
            logger.info("No component results found, running component-specific tests first")
            self.run_component_specific_tests(period_name)
        
        if component_name not in self.component_results:
            logger.error(f"Component {component_name} not found in results")
            return None
        
        base_config = self.component_results[component_name]['config']
        base_params = self._get_base_parameters()
        base_params.update(base_config)
        
        # Ensure position size is capped at 100.0 for safety
        base_params['max_position_size'] = 100.0
        
        # Convert date range to datetime objects
        start_date = datetime.strptime(period_info['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(period_info['end_date'], '%Y-%m-%d')
        
        # Calculate total days in the period
        total_days = (end_date - start_date).days
        fold_size = total_days // folds
        
        # Create folds
        fold_ranges = []
        for i in range(folds):
            fold_start = start_date + timedelta(days=i * fold_size)
            fold_end = start_date + timedelta(days=(i + 1) * fold_size - 1)
            
            # Ensure the last fold includes the end date
            if i == folds - 1:
                fold_end = end_date
            
            fold_ranges.append({
                "fold": i + 1,
                "train_start": start_date.strftime('%Y-%m-%d'),
                "train_end": (fold_start - timedelta(days=1)).strftime('%Y-%m-%d') if i > 0 else None,
                "test_start": fold_start.strftime('%Y-%m-%d'),
                "test_end": fold_end.strftime('%Y-%m-%d')
            })
        
        # Run cross-validation
        fold_results = {}
        
        for fold_range in fold_ranges:
            fold = fold_range["fold"]
            logger.info(f"Running fold {fold} of {folds}")
            
            # Skip first fold train (no data before first fold)
            if fold_range["train_end"] is None:
                logger.info(f"Skipping training for fold {fold} (no training data)")
                continue
            
            # Train on training data
            train_results = self._run_fold_training(
                fold,
                base_params,
                fold_range["train_start"],
                fold_range["train_end"]
            )
            
            # Test on test data
            test_results = self._run_fold_testing(
                fold,
                base_params,
                fold_range["test_start"],
                fold_range["test_end"]
            )
            
            # Store results
            fold_results[fold] = {
                "fold_range": fold_range,
                "train_results": train_results,
                "test_results": test_results,
                "train_metrics": self._calculate_simulation_metrics(train_results),
                "test_metrics": self._calculate_simulation_metrics(test_results)
            }
            
            # Log fold summary
            self._log_fold_summary(fold, fold_range, train_results, test_results)
        
        # Analyze cross-validation results
        analysis = self._analyze_cross_validation_results(fold_results)
        
        # Store results
        cross_validation_results = {
            "component": component_name,
            "period": period_name,
            "folds": folds,
            "fold_results": fold_results,
            "analysis": analysis
        }
        
        self.cross_validation_results[f"{component_name}_{period_name}"] = cross_validation_results
        
        return cross_validation_results
    
    def _run_fold_training(self, fold, params, start_date, end_date):
        """
        Run training for a fold
        
        Args:
            fold (int): Fold number
            params (dict): Parameters for the fold
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            dict: Training results
        """
        logger.info(f"Training fold {fold}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Aggregate results across symbols and timeframes
        aggregate_results = {}
        
        for symbol in self.symbols:
            aggregate_results[symbol] = {}
            
            for timeframe in self.timeframes:
                logger.info(f"Training fold {fold} on {symbol} {timeframe}")
                
                try:
                    # Initialize backtest runner with parameters
                    from ultra_optimized_strategy import UltraOptimizedBacktest
                    backtest = UltraOptimizedBacktest(
                        symbol=symbol, 
                        timeframe=timeframe, 
                        initial_balance=self.initial_balance
                    )
                    
                    # Set date range
                    backtest.strategy.start_date = start_date
                    backtest.strategy.end_date = end_date
                    
                    # Apply parameters
                    backtest.strategy.params.update(params)
                    
                    # Run backtest (training)
                    results = backtest.run()
                    
                    # Store results
                    aggregate_results[symbol][timeframe] = results
                    
                except Exception as e:
                    logger.error(f"Error in training fold {fold} on {symbol} {timeframe}: {str(e)}")
                    aggregate_results[symbol][timeframe] = {"error": str(e)}
        
        return aggregate_results
    
    def _run_fold_testing(self, fold, params, start_date, end_date):
        """
        Run testing for a fold
        
        Args:
            fold (int): Fold number
            params (dict): Parameters for the fold
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            dict: Testing results
        """
        logger.info(f"Testing fold {fold}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Aggregate results across symbols and timeframes
        aggregate_results = {}
        
        for symbol in self.symbols:
            aggregate_results[symbol] = {}
            
            for timeframe in self.timeframes:
                logger.info(f"Testing fold {fold} on {symbol} {timeframe}")
                
                try:
                    # Initialize backtest runner with parameters
                    from ultra_optimized_strategy import UltraOptimizedBacktest
                    backtest = UltraOptimizedBacktest(
                        symbol=symbol, 
                        timeframe=timeframe, 
                        initial_balance=self.initial_balance
                    )
                    
                    # Set date range
                    backtest.strategy.start_date = start_date
                    backtest.strategy.end_date = end_date
                    
                    # Apply parameters
                    backtest.strategy.params.update(params)
                    
                    # Run backtest (testing)
                    results = backtest.run()
                    
                    # Store results
                    aggregate_results[symbol][timeframe] = results
                    
                except Exception as e:
                    logger.error(f"Error in testing fold {fold} on {symbol} {timeframe}: {str(e)}")
                    aggregate_results[symbol][timeframe] = {"error": str(e)}
        
        return aggregate_results
    
    def _log_fold_summary(self, fold, fold_range, train_results, test_results):
        """
        Log summary of fold results
        
        Args:
            fold (int): Fold number
            fold_range (dict): Fold date range
            train_results (dict): Training results
            test_results (dict): Testing results
        """
        logger.info(f"Summary for fold {fold}")
        logger.info(f"Train date range: {fold_range['train_start']} to {fold_range['train_end']}")
        logger.info(f"Test date range: {fold_range['test_start']} to {fold_range['test_end']}")
        
        # Calculate train metrics
        train_metrics = self._calculate_simulation_metrics(train_results)
        
        # Calculate test metrics
        test_metrics = self._calculate_simulation_metrics(test_results)
        
        # Log metrics
        logger.info("\nTraining metrics:")
        logger.info(f"Total profit: ${train_metrics['total_profit']:.2f}")
        logger.info(f"Win rate: {train_metrics['win_rate']*100:.2f}%")
        logger.info(f"Max drawdown: {train_metrics['max_drawdown']*100:.2f}%")
        
        logger.info("\nTesting metrics:")
        logger.info(f"Total profit: ${test_metrics['total_profit']:.2f}")
        logger.info(f"Win rate: {test_metrics['win_rate']*100:.2f}%")
        logger.info(f"Max drawdown: {test_metrics['max_drawdown']*100:.2f}%")
        
        # Calculate performance difference
        profit_diff = test_metrics['total_profit'] - train_metrics['total_profit']
        win_rate_diff = test_metrics['win_rate'] - train_metrics['win_rate']
        drawdown_diff = test_metrics['max_drawdown'] - train_metrics['max_drawdown']
        
        logger.info("\nPerformance difference (test - train):")
        logger.info(f"Profit difference: ${profit_diff:.2f}")
        logger.info(f"Win rate difference: {win_rate_diff*100:.2f}%")
        logger.info(f"Max drawdown difference: {drawdown_diff*100:.2f}%")
        logger.info("-" * 50)
    
    def _analyze_cross_validation_results(self, fold_results):
        """
        Analyze cross-validation results
        
        Args:
            fold_results (dict): Results for each fold
            
        Returns:
            dict: Analysis results
        """
        # Calculate performance consistency across folds
        train_profits = []
        test_profits = []
        train_win_rates = []
        test_win_rates = []
        
        for fold, data in fold_results.items():
            if 'train_metrics' in data and 'test_metrics' in data:
                train_profits.append(data['train_metrics']['total_profit'])
                test_profits.append(data['test_metrics']['total_profit'])
                train_win_rates.append(data['train_metrics']['win_rate'])
                test_win_rates.append(data['test_metrics']['win_rate'])
        
        # Calculate mean and standard deviation
        train_profit_mean = np.mean(train_profits) if train_profits else 0
        train_profit_std = np.std(train_profits) if train_profits else 0
        test_profit_mean = np.mean(test_profits) if test_profits else 0
        test_profit_std = np.std(test_profits) if test_profits else 0
        
        train_win_rate_mean = np.mean(train_win_rates) if train_win_rates else 0
        train_win_rate_std = np.std(train_win_rates) if train_win_rates else 0
        test_win_rate_mean = np.mean(test_win_rates) if test_win_rates else 0
        test_win_rate_std = np.std(test_win_rates) if test_win_rates else 0
        
        # Calculate coefficient of variation (lower is better)
        train_profit_cv = train_profit_std / train_profit_mean if train_profit_mean != 0 else float('inf')
        test_profit_cv = test_profit_std / test_profit_mean if test_profit_mean != 0 else float('inf')
        
        # Calculate overfitting score (lower is better)
        # Based on the difference between train and test performance
        profit_overfitting = (train_profit_mean - test_profit_mean) / train_profit_mean if train_profit_mean != 0 else 0
        win_rate_overfitting = train_win_rate_mean - test_win_rate_mean
        
        # Calculate consistency score (higher is better)
        # Based on the coefficient of variation of test performance
        consistency_score = max(0, 100 - test_profit_cv * 100)
        
        return {
            "train_profit_mean": train_profit_mean,
            "train_profit_std": train_profit_std,
            "test_profit_mean": test_profit_mean,
            "test_profit_std": test_profit_std,
            "train_win_rate_mean": train_win_rate_mean,
            "train_win_rate_std": train_win_rate_std,
            "test_win_rate_mean": test_win_rate_mean,
            "test_win_rate_std": test_win_rate_std,
            "profit_overfitting": profit_overfitting,
            "win_rate_overfitting": win_rate_overfitting,
            "consistency_score": consistency_score,
            "is_overfitting": profit_overfitting > 0.2 or win_rate_overfitting > 0.1  # Thresholds for overfitting
        }
    
    def _log_cross_validation_summary(self, cross_validation_results):
        """
        Log summary of cross-validation results
        
        Args:
            cross_validation_results (dict): Cross-validation results
        """
        analysis = cross_validation_results['analysis']
        
        logger.info(f"Cross-Validation Summary ({cross_validation_results['folds']} folds)")
        logger.info(f"Component: {cross_validation_results['component']}")
        logger.info(f"Period: {cross_validation_results['period']}")
        
        logger.info("\nTraining Performance:")
        logger.info(f"Mean profit: ${analysis['train_profit_mean']:.2f}")
        logger.info(f"Standard deviation: ${analysis['train_profit_std']:.2f}")
        logger.info(f"Mean win rate: {analysis['train_win_rate_mean']*100:.2f}%")
        logger.info(f"Standard deviation: {analysis['train_win_rate_std']*100:.2f}%")
        
        logger.info("\nTesting Performance:")
        logger.info(f"Mean profit: ${analysis['test_profit_mean']:.2f}")
        logger.info(f"Standard deviation: ${analysis['test_profit_std']:.2f}")
        logger.info(f"Mean win rate: {analysis['test_win_rate_mean']*100:.2f}%")
        logger.info(f"Standard deviation: {analysis['test_win_rate_std']*100:.2f}%")
        
        logger.info("\nOverfitting Analysis:")
        logger.info(f"Profit overfitting: {analysis['profit_overfitting']*100:.2f}%")
        logger.info(f"Win rate overfitting: {analysis['win_rate_overfitting']*100:.2f}%")
        logger.info(f"Consistency score: {analysis['consistency_score']:.2f}")
        logger.info(f"Is overfitting: {analysis['is_overfitting']}")
        logger.info("-" * 50)


class AdvancedComponentBacktest(AdvancedBacktestFramework):
    """
    Advanced component backtesting class that extends the AdvancedBacktestFramework
    with additional functionality for testing specific components of the trading strategy.
    
    This class is designed to be used with the main.py entry point.
    """
    
    def __init__(self, symbols=None, timeframes=None, initial_balance=50000):
        """
        Initialize the advanced component backtest
        
        Args:
            symbols (list): List of symbols to backtest
            timeframes (list): List of timeframes to backtest
            initial_balance (float): Initial balance
        """
        super().__init__(symbols, timeframes, initial_balance)
        
        # Ensure position sizes are capped at 100.0 for safety
        self.max_position_size = 100.0
        
        logger.info("Advanced Component Backtest initialized")
        logger.info(f"Maximum position size capped at {self.max_position_size}")
        
    def run_component_specific_tests(self, period="1_year"):
        """
        Run tests for specific components of the trading strategy
        
        Args:
            period (str): Period to test (1_month, 6_months, 1_year, 5_years)
            
        Returns:
            dict: Results for each component
        """
        logger.info(f"Running component-specific tests for period: {period}")
        
        # Run the component tests from the parent class
        results = super().run_component_specific_tests(period)
        
        # Save results to file
        self._save_component_results(results, period)
        
        return results
    
    def run_monte_carlo_simulations(self, period="1_year", component="full_ensemble", simulation_count=1000):
        """
        Run Monte Carlo simulations with parameter randomization
        
        Args:
            period (str): Period to test
            component (str): Component to test
            simulation_count (int): Number of simulations to run
            
        Returns:
            dict: Monte Carlo simulation results
        """
        logger.info(f"Running Monte Carlo simulations for period: {period}, component: {component}")
        
        # Run the Monte Carlo simulations from the parent class
        results = super().run_monte_carlo_simulations(period, simulation_count, component)
        
        # Save results to file
        self._save_monte_carlo_results(results, period, component)
        
        return results
    
    def run_stress_tests(self, period="1_year", component="full_ensemble"):
        """
        Run stress tests to evaluate strategy performance under extreme market conditions
        
        Args:
            period (str): Period to test
            component (str): Component to test
            
        Returns:
            dict: Stress test results
        """
        logger.info(f"Running stress tests for period: {period}, component: {component}")
        
        # Run the stress tests from the parent class
        results = super().run_stress_tests(period, component)
        
        # Save results to file
        self._save_stress_test_results(results, period, component)
        
        return results
    
    def run_cross_validation(self, period="1_year", component="full_ensemble", folds=5):
        """
        Run cross-validation to prevent overfitting
        
        Args:
            period (str): Period to test
            component (str): Component to test
            folds (int): Number of folds for cross-validation
            
        Returns:
            dict: Cross-validation results
        """
        logger.info(f"Running cross-validation for period: {period}, component: {component}")
        
        # Run the cross-validation from the parent class
        results = super().run_cross_validation(period, component, folds)
        
        # Save results to file
        self._save_cross_validation_results(results, period, component)
        
        return results
    
    def _save_component_results(self, results, period):
        """
        Save component test results to file
        
        Args:
            results (dict): Component test results
            period (str): Period tested
        """
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs("results/backtest_results", exist_ok=True)
        
        # Save results to file
        filename = f"results/backtest_results/component_tests_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Component test results saved to {filename}")
    
    def _save_monte_carlo_results(self, results, period, component):
        """
        Save Monte Carlo simulation results to file
        
        Args:
            results (dict): Monte Carlo simulation results
            period (str): Period tested
            component (str): Component tested
        """
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs("results/monte_carlo_results", exist_ok=True)
        
        # Save results to file
        filename = f"results/monte_carlo_results/monte_carlo_{component}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Monte Carlo simulation results saved to {filename}")
    
    def _save_stress_test_results(self, results, period, component):
        """
        Save stress test results to file
        
        Args:
            results (dict): Stress test results
            period (str): Period tested
            component (str): Component tested
        """
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs("results/stress_test_results", exist_ok=True)
        
        # Save results to file
        filename = f"results/stress_test_results/stress_tests_{component}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Stress test results saved to {filename}")
    
    def _save_cross_validation_results(self, results, period, component):
        """
        Save cross-validation results to file
        
        Args:
            results (dict): Cross-validation results
            period (str): Period tested
            component (str): Component tested
        """
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs("results/cross_validation_results", exist_ok=True)
        
        # Save results to file
        filename = f"results/cross_validation_results/cross_validation_{component}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Cross-validation results saved to {filename}")
