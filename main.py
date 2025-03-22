"""
Main entry point for the trading system.
This file provides a command-line interface to run backtests, 
stress tests, and live trading.
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/trading_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def load_config(config_path="config/trading_parameters.json"):
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)

def run_backtest(args, config):
    """
    Run a backtest with the specified parameters.
    
    Args:
        args (argparse.Namespace): Command-line arguments
        config (dict): Configuration parameters
    """
    logger.info("Starting backtest")
    
    # Import here to avoid circular imports
    from backtest.advanced_component_backtest import AdvancedComponentBacktest
    
    # Get symbols and timeframes
    symbols = args.symbols.split(',') if args.symbols else config['symbols']
    timeframes = args.timeframes.split(',') if args.timeframes else config['timeframes']
    
    # Get period
    period = args.period if args.period else "1_year"
    
    # Get component
    component = args.component if args.component else "full_ensemble"
    
    # Create backtest instance
    backtest = AdvancedComponentBacktest(
        symbols=symbols,
        timeframes=timeframes,
        initial_balance=config['general']['initial_balance']
    )
    
    # Run component-specific tests
    if args.component_tests:
        logger.info("Running component-specific tests")
        results = backtest.run_component_specific_tests(period)
        logger.info(f"Component test results: {results}")
    
    # Run Monte Carlo simulations
    if args.monte_carlo:
        logger.info("Running Monte Carlo simulations")
        simulation_count = args.simulation_count if args.simulation_count else config['backtest_settings']['monte_carlo']['simulation_count']
        results = backtest.run_monte_carlo_simulations(period, component, simulation_count)
        logger.info(f"Monte Carlo simulation results: {results}")
    
    # Run stress tests
    if args.stress_tests:
        logger.info("Running stress tests")
        results = backtest.run_stress_tests(period, component)
        logger.info(f"Stress test results: {results}")
    
    # Run cross-validation
    if args.cross_validation:
        logger.info("Running cross-validation")
        folds = args.folds if args.folds else config['backtest_settings']['cross_validation']['folds']
        results = backtest.run_cross_validation(period, component, folds)
        logger.info(f"Cross-validation results: {results}")
    
    logger.info("Backtest completed")

def run_live_trading(args, config):
    """
    Run live trading with the specified parameters.
    
    Args:
        args (argparse.Namespace): Command-line arguments
        config (dict): Configuration parameters
    """
    logger.info("Starting live trading")
    
    # Import here to avoid circular imports
    from live.mt5_live_trading import MT5LiveTrader
    
    # Get symbols and timeframes
    symbols = args.symbols.split(',') if args.symbols else config['symbols']
    timeframes = args.timeframes.split(',') if args.timeframes else config['timeframes']
    
    # Create live trader instance
    trader = MT5LiveTrader(config_file=args.config_file)
    
    # Start trading
    trader.start_trading(symbols, timeframes)
    
    logger.info("Live trading started")

def main():
    """
    Main function to parse command-line arguments and run the appropriate mode.
    """
    parser = argparse.ArgumentParser(description="Trading System CLI")
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['backtest', 'live'], required=True,
                        help='Mode to run: backtest or live trading')
    
    # General parameters
    parser.add_argument('--config-file', type=str, default='config/trading_parameters.json',
                        help='Path to configuration file')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to trade')
    parser.add_argument('--timeframes', type=str, help='Comma-separated list of timeframes to trade')
    
    # Backtest parameters
    parser.add_argument('--period', type=str, choices=['1_month', '6_months', '1_year', '5_years'],
                        help='Period to backtest')
    parser.add_argument('--component', type=str, help='Component to test')
    parser.add_argument('--component-tests', action='store_true', help='Run component-specific tests')
    parser.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo simulations')
    parser.add_argument('--simulation-count', type=int, help='Number of Monte Carlo simulations')
    parser.add_argument('--stress-tests', action='store_true', help='Run stress tests')
    parser.add_argument('--cross-validation', action='store_true', help='Run cross-validation')
    parser.add_argument('--folds', type=int, help='Number of cross-validation folds')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Run appropriate mode
    if args.mode == 'backtest':
        run_backtest(args, config)
    elif args.mode == 'live':
        run_live_trading(args, config)

if __name__ == "__main__":
    main()
