from .ultra_optimized_backtest import UltraOptimizedBacktest

def run_comprehensive_backtest(symbols=["BTCUSDT", "ETHUSDT"], 
                             timeframes=["3m", "5m", "15m"], 
                             initial_balance=50000,
                             start_date="2024-11-01", 
                             end_date="2025-02-28",
                             optimization_rounds=2):
    """
    Run a comprehensive backtest across multiple symbols and timeframes
    
    Args:
        symbols (list): List of symbols to backtest
        timeframes (list): List of timeframes to backtest
        initial_balance (float): Initial balance
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        optimization_rounds (int): Number of optimization rounds to perform
        
    Returns:
        dict: Aggregated backtest results by symbol and timeframe
    """
    logger.info(f"Running comprehensive backtest for {len(symbols)} symbols on {len(timeframes)} timeframes")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Optimization rounds: {optimization_rounds}")
    
    # Store results for each symbol and timeframe
    all_results = {}
    historical_performance = {}
    
    # Track best performing configurations
    best_profit_factor = 0
    best_win_rate = 0
    best_avg_win = 0
    best_config = {}
    
    # Run multiple optimization rounds
    for round_num in range(1, optimization_rounds + 1):
        logger.info(f"Starting optimization round {round_num} of {optimization_rounds}")
        
        # Run backtest for each symbol and timeframe
        for symbol in symbols:
            if symbol not in all_results:
                all_results[symbol] = {}
                historical_performance[symbol] = {}
            
            for timeframe in timeframes:
                logger.info(f"Running backtest for {symbol} on {timeframe} timeframe (Round {round_num})")
                
                try:
                    # Initialize backtest runner
                    backtest = UltraOptimizedBacktest(symbol=symbol, timeframe=timeframe, initial_balance=initial_balance)
                    
                    # Set date range
                    backtest.strategy.start_date = start_date
                    backtest.strategy.end_date = end_date
                    
                    # Get optimal parameters based on historical performance (if available)
                    previous_performance = historical_performance[symbol].get(timeframe, None)
                    optimal_params = generate_optimal_parameters(symbol, timeframe, previous_performance)
                    
                    # Apply additional round-specific adjustments
                    if round_num > 1:
                        # If we're in a subsequent round, make more aggressive adjustments
                        if previous_performance:
                            if previous_performance['win_rate'] < 0.80:
                                # If win rate is still below target, make more selective
                                optimal_params['confidence_threshold'] += 0.01 * (round_num - 1)
                                optimal_params['force_signals'] = True
                                optimal_params['force_signal_threshold'] += 0.02 * (round_num - 1)
                            
                            if previous_performance['profit_factor'] < 50.0:
                                # If profit factor is still below target, be more aggressive
                                optimal_params['position_size_multiplier'] += 2.5 * (round_num - 1)
                                optimal_params['take_profit_pct'] += 0.02 * (round_num - 1)
                            
                            if previous_performance['average_win_pct'] < 0.20:
                                # If average win is still below target, adjust profit taking
                                optimal_params['take_profit_pct'] += 0.02 * (round_num - 1)
                                optimal_params['trailing_stop_activation'] -= 0.01 * (round_num - 1)
                    
                    # Ensure position size is capped at 100.0 for live trading safety
                    optimal_params['max_position_size'] = 100.0
                    
                    # Set ultra-optimized parameters
                    backtest.strategy.params.update(optimal_params)
                    
                    # Set advanced quantitative components with ultra-optimized parameters
                    backtest.strategy.params.update({
                        # Advanced components
                        'use_markov_chains': True,         # Use Markov chain models
                        'use_ml_models': True,             # Use ML models
                        'use_fractal_analysis': True,      # Use fractal analysis
                        'use_microstructure': True,        # Use microstructure noise analysis
                        'use_volume_analysis': True,       # Use volume analysis
                        'use_rnn_strategy_selection': True, # Use RNN for strategy selection
                        'use_bid_ask_imbalance': True,     # Use bid/ask volume imbalance
                        
                        # Component weights for ensemble confidence - optimized for higher quality signals
                        'markov_confidence_weight': 0.30,   # Weight for Markov models
                        'ml_confidence_weight': 0.25,       # Weight for ML models
                        'fractal_confidence_weight': 0.20,  # Weight for fractal analysis
                        'price_action_weight': 0.15,        # Weight for price action
                        'volume_confidence_weight': 0.10,   # Weight for volume analysis
                        
                        # Timeframe-specific adjustments
                        'timeframe_multiplier': 1.0 if timeframe == "5m" else (0.8 if timeframe == "3m" else 1.2),
                        
                        # Signal generation parameters - ultra-optimized for higher win rate
                        'confidence_threshold': 0.90,       # Increased threshold for higher quality signals
                        'force_signals': True,              # Force signals if no trades for a while
                        'confidence_decay': 0.97,           # Slower decay for more selective signals
                        'min_signals_per_day': 3 if timeframe == "3m" else (2 if timeframe == "5m" else 1),
                        'max_signals_per_day': 8,           # Limit signals per day for quality
                        'opportunity_scan_interval': 3,     # Scan for opportunities more frequently
                        'force_signal_threshold': 0.85,     # Higher threshold for forced signals
                        
                        # Position sizing and risk management - ultra-aggressive but capped for live trading
                        'ultra_aggressive': True,           # Enable ultra-aggressive mode
                        'position_size_multiplier': 50.0,   # Significantly increased multiplier for higher returns
                        'max_position_size': 100.0,         # Cap position size for live trading safety
                        'initial_risk_per_trade': 0.005,    # Reduced initial risk per trade (0.5%)
                        'win_streak_multiplier': 1.5,       # Increased multiplier after wins
                        'loss_streak_divider': 2.0,         # Increased divider after losses
                        'max_risk_multiplier': 5.0,         # Maximum risk multiplier
                        'adaptive_position_sizing': True,   # Use adaptive position sizing
                        
                        # Trade management - optimized for higher profit targets
                        'stop_loss_pct': 0.03 if timeframe == "3m" else (0.04 if timeframe == "5m" else 0.05),
                        'take_profit_pct': 0.25 if timeframe == "3m" else (0.30 if timeframe == "5m" else 0.35),
                        'max_concurrent_trades': 5,         # Reduced max concurrent trades for higher quality
                        'max_trade_duration': 60 if timeframe == "3m" else (100 if timeframe == "5m" else 150),
                        
                        # Partial profit taking
                        'use_partial_exits': True,          # Enable partial profit taking
                        'partial_exit_levels': [0.15, 0.25, 0.35], # Take partial profits at these levels
                        'partial_exit_percentages': [0.3, 0.3, 0.4], # Percentage of position to close at each level
                        
                        # Trailing stop parameters - optimized for capturing larger moves
                        'use_trailing_stop': True,          # Use trailing stops
                        'trailing_stop_activation': 0.10 if timeframe == "3m" else (0.12 if timeframe == "5m" else 0.15),
                        'trailing_stop_distance': 0.05 if timeframe == "3m" else (0.06 if timeframe == "5m" else 0.08),
                        'adaptive_trailing_stop': True,     # Use adaptive trailing stops based on volatility
                        'volatility_trailing_multiplier': 1.5, # Multiplier for volatility-based trailing stops
                        
                        # Advanced ML parameters
                        'ml_lookback_periods': 500,         # Increased lookback for ML models
                        'ml_prediction_threshold': 0.80,    # Higher threshold for ML predictions
                        'ml_ensemble_size': 15,             # Increased ensemble size for ML models
                        'ml_feature_engineering': True,     # Enable advanced feature engineering
                        'ml_rescoring_enabled': True,       # Enable ML rescoring of signals
                        'random_forest_depth': 10,          # Depth for Random Forest models
                        'random_forest_estimators': 100,    # Number of estimators for Random Forest
                        
                        # RNN strategy selection parameters
                        'rnn_lookback': 200,                # Lookback period for RNN
                        'rnn_hidden_layers': 3,             # Number of hidden layers
                        'rnn_neurons_per_layer': 64,        # Neurons per layer
                        'rnn_dropout': 0.2,                 # Dropout rate
                        'rnn_batch_size': 32,               # Batch size for training
                        
                        # Markov chain parameters
                        'markov_state_count': 7,            # Increased number of states in Markov model
                        'markov_lookback': 250,             # Increased lookback period for Markov models
                        'markov_entropy_threshold': 0.7,    # Higher entropy threshold for Markov signals
                        'markov_transition_smoothing': 0.1, # Smoothing factor for transition probabilities
                        
                        # Fractal analysis parameters
                        'hurst_window': 120,                # Increased window for Hurst exponent calculation
                        'hurst_trending_threshold': 0.65,   # Increased threshold for trending detection
                        'hurst_ranging_threshold': 0.40,    # Decreased threshold for ranging detection
                        'fractal_dim_threshold': 1.4,       # Adjusted threshold for fractal dimension
                        'fractal_window_sizes': [20, 50, 100], # Multiple window sizes for fractal analysis
                        
                        # Microstructure noise parameters
                        'noise_window': 50,                 # Window for noise calculation
                        'noise_threshold_multiplier': 2.0,  # Increased threshold for noise detection
                        'noise_filtering_enabled': True,    # Enable noise filtering
                        
                        # Volume analysis parameters
                        'volume_lookback': 100,             # Lookback period for volume analysis
                        'volume_ma_periods': [10, 20, 50],  # Moving average periods for volume
                        'volume_threshold': 1.5,            # Volume threshold for signals
                        'bid_ask_imbalance_threshold': 2.0, # Threshold for bid/ask imbalance
                        
                        # RSI parameters
                        'rsi_period': 14,                   # RSI period
                        'rsi_overbought': 70,               # RSI overbought level
                        'rsi_oversold': 30,                 # RSI oversold level
                        'rsi_extreme_threshold': 0.1,       # Extreme RSI threshold for ultra-aggressive mode
                        
                        # Market regime detection
                        'regime_detection_enabled': True,   # Enable market regime detection
                        'regime_lookback': 500,             # Lookback period for regime detection
                        'trend_strength_threshold': 0.6,    # Threshold for trend strength
                        
                        # Signal reversal parameters
                        'check_signal_reversal': True,      # Check for signal reversals
                        'reversal_threshold': 0.8,          # Threshold for signal reversals
                        
                        # Target metrics
                        'target_avg_return': 0.20,          # Target average return per trade (20%)
                        'target_win_rate': 0.80,            # Target win rate (80%)
                        'target_profit_factor': 50.0,       # Target profit factor (50)
                        
                        # Debug mode
                        'debug_mode': False,                # Disable debug mode for faster execution
                    })
                    
                    # Run backtest
                    results = backtest.run_backtest()
                    
                    if results:
                        all_results[symbol][timeframe] = results
                        
                        # Store equity curve for combined analysis
                        if 'equity_curve' in results and 'dates' in results:
                            # Normalize equity curve to start from initial_balance
                            normalized_equity = [val / results['equity_curve'][0] * initial_balance for val in results['equity_curve']]
                            combined_equity = [(timeframe, normalized_equity, results['dates'])]
                        
                        # Store historical performance
                        historical_performance[symbol][timeframe] = {
                            'win_rate': results['win_rate'],
                            'profit_factor': results['profit_factor'],
                            'average_win_pct': results['average_win_pct']
                        }
                        
                        # Check if this configuration outperforms others
                        if results['profit_factor'] > best_profit_factor:
                            best_profit_factor = results['profit_factor']
                            best_config['profit_factor'] = {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'profit_factor': results['profit_factor'],
                                'win_rate': results['win_rate'],
                                'avg_win': results['average_win_pct'],
                                'parameters': optimal_params,
                                'optimization_round': round_num
                            }
                        
                        if results['win_rate'] > best_win_rate:
                            best_win_rate = results['win_rate']
                            best_config['win_rate'] = {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'profit_factor': results['profit_factor'],
                                'win_rate': results['win_rate'],
                                'avg_win': results['average_win_pct'],
                                'parameters': optimal_params,
                                'optimization_round': round_num
                            }
                        
                        if results['average_win_pct'] > best_avg_win:
                            best_avg_win = results['average_win_pct']
                            best_config['avg_win'] = {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'profit_factor': results['profit_factor'],
                                'win_rate': results['win_rate'],
                                'avg_win': results['average_win_pct'],
                                'parameters': optimal_params,
                                'optimization_round': round_num
                            }
                
                except Exception as e:
                    logger.error(f"Error running backtest for {symbol} on {timeframe} (Round {round_num}): {e}")
                    logger.error(f"Skipping {symbol} on {timeframe} for this round")
                    continue
        
        # After each round, log progress
        logger.info(f"Completed optimization round {round_num} of {optimization_rounds}")
        logger.info(f"Current best profit factor: {best_profit_factor:.2f}")
        logger.info(f"Current best win rate: {best_win_rate*100:.2f}%")
        logger.info(f"Current best average win: {best_avg_win*100:.2f}%")
    
    # Print comprehensive results summary
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTEST RESULTS SUMMARY")
    print("="*80)
    
    # Calculate and print aggregated metrics by symbol
    for symbol in symbols:
        if symbol in all_results and all_results[symbol]:
            try:
                symbol_results = all_results[symbol]
                total_trades = sum(results['total_trades'] for results in symbol_results.values())
                total_winning = sum(results['winning_trades'] for results in symbol_results.values())
                total_profit = sum(results['net_profit'] for results in symbol_results.values())
                
                # Calculate aggregated metrics
                agg_win_rate = total_winning / total_trades if total_trades > 0 else 0
                agg_profit_factor = sum(results['profit_factor'] * results['total_trades'] for results in symbol_results.values()) / total_trades if total_trades > 0 else 0
                agg_avg_win = sum(results['average_win_pct'] * results['winning_trades'] for results in symbol_results.values()) / total_winning if total_winning > 0 else 0
                
                print(f"\nSymbol: {symbol}")
                print(f"Date Range: {start_date} to {end_date}")
                print(f"Initial Balance: ${initial_balance:.2f}")
                print(f"Final Balance: ${initial_balance + total_profit:.2f}")
                print(f"Net Profit: ${total_profit:.2f} ({total_profit/initial_balance*100:.2f}%)")
                print(f"Total Trades: {total_trades}")
                print(f"Aggregated Win Rate: {agg_win_rate*100:.2f}%")
                print(f"Aggregated Profit Factor: {agg_profit_factor:.2f}")
                print(f"Aggregated Average Win: {agg_avg_win*100:.2f}%")
                
                # Print results for each timeframe
                print("\nTimeframe Results:")
                for timeframe, results in symbol_results.items():
                    print(f"  {timeframe}: Win Rate: {results['win_rate']*100:.2f}%, Profit Factor: {results['profit_factor']:.2f}, Avg Win: {results['average_win_pct']*100:.2f}%")
            
            except Exception as e:
                logger.error(f"Error calculating aggregated metrics for {symbol}: {e}")
                continue
    
    # Print best configurations
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)
    
    if 'profit_factor' in best_config:
        print("\nBest Profit Factor Configuration:")
        print(f"Symbol: {best_config['profit_factor']['symbol']}")
        print(f"Timeframe: {best_config['profit_factor']['timeframe']}")
        print(f"Optimization Round: {best_config['profit_factor']['optimization_round']}")
        print(f"Profit Factor: {best_config['profit_factor']['profit_factor']:.2f}")
        print(f"Win Rate: {best_config['profit_factor']['win_rate']*100:.2f}%")
        print(f"Average Win: {best_config['profit_factor']['avg_win']*100:.2f}%")
        
        # Print key parameters
        params = best_config['profit_factor']['parameters']
        print("\nKey Parameters:")
        print(f"  Confidence Threshold: {params['confidence_threshold']:.2f}")
        print(f"  Position Size Multiplier: {params['position_size_multiplier']:.1f}")
        print(f"  Take Profit: {params['take_profit_pct']*100:.1f}%")
        print(f"  Stop Loss: {params['stop_loss_pct']*100:.1f}%")
        print(f"  Trailing Stop Activation: {params['trailing_stop_activation']*100:.1f}%")
        print(f"  Trailing Stop Distance: {params['trailing_stop_distance']*100:.1f}%")
    
    if 'win_rate' in best_config:
        print("\nBest Win Rate Configuration:")
        print(f"Symbol: {best_config['win_rate']['symbol']}")
        print(f"Timeframe: {best_config['win_rate']['timeframe']}")
        print(f"Optimization Round: {best_config['win_rate']['optimization_round']}")
        print(f"Profit Factor: {best_config['win_rate']['profit_factor']:.2f}")
        print(f"Win Rate: {best_config['win_rate']['win_rate']*100:.2f}%")
        print(f"Average Win: {best_config['win_rate']['avg_win']*100:.2f}%")
    
    if 'avg_win' in best_config:
        print("\nBest Average Win Configuration:")
        print(f"Symbol: {best_config['avg_win']['symbol']}")
        print(f"Timeframe: {best_config['avg_win']['timeframe']}")
        print(f"Optimization Round: {best_config['avg_win']['optimization_round']}")
        print(f"Profit Factor: {best_config['avg_win']['profit_factor']:.2f}")
        print(f"Win Rate: {best_config['avg_win']['win_rate']*100:.2f}%")
        print(f"Average Win: {best_config['avg_win']['avg_win']*100:.2f}%")
    
    # Check if target metrics are achieved
    target_achieved = False
    for symbol in symbols:
        if symbol in all_results:
            for timeframe, results in all_results[symbol].items():
                if (results['profit_factor'] >= 50.0 and 
                    results['win_rate'] >= 0.80 and 
                    results['average_win_pct'] >= 0.20):
                    target_achieved = True
                    print("\n" + "="*80)
                    print("TARGET METRICS ACHIEVED!")
                    print("="*80)
                    print(f"Symbol: {symbol}")
                    print(f"Timeframe: {timeframe}")
                    print(f"Profit Factor: {results['profit_factor']:.2f}")
                    print(f"Win Rate: {results['win_rate']*100:.2f}%")
                    print(f"Average Win: {results['average_win_pct']*100:.2f}%")
                    break
            if target_achieved:
                break
    
    if not target_achieved:
        print("\n" + "="*80)
        print("TARGET METRICS NOT ACHIEVED")
        print("="*80)
        print("Consider further optimization of strategy parameters or increasing the number of optimization rounds.")
    
    return all_results, best_config

def generate_optimal_parameters(symbol, timeframe, historical_performance=None):
    """
    Generate optimal strategy parameters based on symbol, timeframe, and historical performance
    
    Args:
        symbol (str): Symbol to generate parameters for
        timeframe (str): Timeframe to generate parameters for
        historical_performance (dict): Historical performance data from previous backtest
        
    Returns:
        dict: Optimal parameters
    """
    logger.info(f"Generating optimal parameters for {symbol} on {timeframe} timeframe")
    
    # Base parameters - these will be adjusted based on symbol, timeframe, and historical performance
    params = {
        # Position sizing and risk management
        'position_size_multiplier': 10.0,   # Conservative multiplier
        'max_position_size': 100.0,         # Cap position size for safety
        'initial_risk_per_trade': 0.01,     # Conservative risk per trade (1%)
        'win_streak_multiplier': 1.2,       # Increase size after wins
        'loss_streak_divider': 1.5,         # Decrease size after losses
        'max_risk_multiplier': 3.0,         # Maximum risk multiplier
        'adaptive_position_sizing': True,   # Use adaptive position sizing
        
        # Trade management
        'stop_loss_pct': 0.05,              # Stop loss percentage (5%)
        'take_profit_pct': 0.20,            # Take profit percentage (20%)
        'max_concurrent_trades': 5,         # Max concurrent trades
        'max_trade_duration': 100,          # Maximum trade duration in minutes
        
        # Signal generation parameters
        'confidence_threshold': 0.85,       # Confidence threshold
        'force_signals': True,              # Force signal generation for testing
        'force_signal_threshold': 0.80,     # Threshold for forced signals
        'min_signals_per_day': 3,           # Minimum signals per day
        'max_signals_per_day': 8,           # Maximum signals per day
        
        # Advanced components
        'use_markov_chains': True,          # Use Markov chain models
        'use_ml_models': True,              # Use ML models
        'use_fractal_analysis': True,       # Use fractal analysis
        'use_microstructure': True,         # Use microstructure noise analysis
        'use_volume_analysis': True,        # Use volume analysis
        'use_rnn_strategy_selection': True, # Use RNN for strategy selection
        'use_bid_ask_imbalance': True,      # Use bid/ask volume imbalance
        
        # Component weights for ensemble confidence
        'markov_confidence_weight': 0.30,   # Weight for Markov models
        'ml_confidence_weight': 0.25,       # Weight for ML models
        'fractal_confidence_weight': 0.20,  # Weight for fractal analysis
        'price_action_weight': 0.15,        # Weight for price action
        'volume_confidence_weight': 0.10,   # Weight for volume analysis
        
        # RSI parameters
        'rsi_period': 14,                   # RSI period
        'rsi_overbought': 70,               # RSI overbought level
        'rsi_oversold': 30,                 # RSI oversold level
        'rsi_extreme_threshold': 0.1,       # Extreme RSI threshold for ultra-aggressive mode
    }
    
    # Symbol-specific adjustments
    if symbol == "BTCUSDT":
        # Bitcoin tends to have higher volatility
        params.update({
            'stop_loss_pct': 0.06,          # Wider stop loss for BTC
            'take_profit_pct': 0.25,        # Higher take profit for BTC
            'position_size_multiplier': 8.0, # More conservative multiplier for BTC
            'markov_state_count': 8,        # More states for BTC
            'hurst_trending_threshold': 0.62, # Higher trending threshold for BTC
            'noise_threshold_multiplier': 1.8, # Higher noise threshold for BTC
        })
    elif symbol == "ETHUSDT":
        # Ethereum parameters
        params.update({
            'stop_loss_pct': 0.055,         # Slightly wider stop loss for ETH
            'take_profit_pct': 0.22,        # Higher take profit for ETH
            'position_size_multiplier': 9.0, # Slightly more conservative multiplier for ETH
            'markov_state_count': 7,        # Fewer states for ETH
            'hurst_trending_threshold': 0.60, # Lower trending threshold for ETH
            'noise_threshold_multiplier': 1.6, # Lower noise threshold for ETH
        })
    else:
        # Default for other symbols
        params.update({
            'stop_loss_pct': 0.05,          # Default stop loss
            'take_profit_pct': 0.20,        # Default take profit
            'position_size_multiplier': 10.0, # Default multiplier
            'markov_state_count': 6,        # Default states
            'hurst_trending_threshold': 0.58, # Default trending threshold
            'noise_threshold_multiplier': 1.5, # Default noise threshold
        })
    
    # Timeframe-specific adjustments
    if timeframe == "3m":
        # Shorter timeframe - more trades, smaller targets
        params.update({
            'stop_loss_pct': params['stop_loss_pct'] * 0.8,  # Tighter stop loss
            'take_profit_pct': params['take_profit_pct'] * 0.8, # Lower take profit
            'max_concurrent_trades': 8,     # More concurrent trades
            'confidence_threshold': 0.88,   # Higher confidence threshold
            'min_signals_per_day': 5,       # More signals per day
            'max_signals_per_day': 12,      # More signals per day
            'max_trade_duration': 60,       # Shorter trade duration
            'ml_lookback_periods': 300,     # Shorter lookback for ML
            'markov_lookback': 150,         # Shorter lookback for Markov
            'hurst_window': 80,             # Shorter window for Hurst
            'fractal_window_sizes': [10, 30, 60], # Smaller window sizes
            'noise_window': 30,             # Smaller window for noise
            'rnn_lookback': 120,            # Shorter lookback for RNN
        })
    elif timeframe == "5m":
        # Medium timeframe - balanced approach
        params.update({
            'stop_loss_pct': params['stop_loss_pct'] * 0.9,  # Slightly tighter stop loss
            'take_profit_pct': params['take_profit_pct'] * 0.9, # Slightly lower take profit
            'max_concurrent_trades': 6,     # Moderate concurrent trades
            'confidence_threshold': 0.86,   # Moderate confidence threshold
            'min_signals_per_day': 3,       # Moderate signals per day
            'max_signals_per_day': 8,       # Moderate signals per day
            'max_trade_duration': 100,      # Moderate trade duration
            'ml_lookback_periods': 400,     # Moderate lookback for ML
            'markov_lookback': 200,         # Moderate lookback for Markov
            'hurst_window': 100,            # Moderate window for Hurst
            'fractal_window_sizes': [15, 40, 80], # Moderate window sizes
            'noise_window': 40,             # Moderate window for noise
            'rnn_lookback': 160,            # Moderate lookback for RNN
        })
    else:
        # Longer timeframe - fewer trades, larger targets
        params.update({
            'stop_loss_pct': params['stop_loss_pct'] * 1.1,  # Wider stop loss
            'take_profit_pct': params['take_profit_pct'] * 1.1, # Higher take profit
            'max_concurrent_trades': 4,     # Fewer concurrent trades
            'confidence_threshold': 0.84,   # Lower confidence threshold
            'min_signals_per_day': 1,       # Fewer signals per day
            'max_signals_per_day': 5,       # Fewer signals per day
            'max_trade_duration': 150,      # Longer trade duration
            'ml_lookback_periods': 500,     # Longer lookback for ML
            'markov_lookback': 250,         # Longer lookback for Markov
            'hurst_window': 120,            # Longer window for Hurst
            'fractal_window_sizes': [20, 50, 100], # Larger window sizes
            'noise_window': 50,             # Larger window for noise
            'rnn_lookback': 200,            # Longer lookback for RNN
        })
    
    # Adjust based on historical performance if available
    if historical_performance:
        logger.info("Adjusting parameters based on historical performance")
        
        # Get key metrics
        win_rate = historical_performance.get('win_rate', 0.5)
        profit_factor = historical_performance.get('profit_factor', 1.0)
        avg_win_pct = historical_performance.get('average_win_pct', 0.1)
        avg_loss_pct = historical_performance.get('average_loss_pct', -0.05)
        total_trades = historical_performance.get('total_trades', 0)
        
        # Adjust position sizing based on performance
        if profit_factor > 30.0 and win_rate > 0.75:
            # Excellent performance - be more aggressive
            params['position_size_multiplier'] = min(params['position_size_multiplier'] * 1.5, 50.0)
            params['max_position_size'] = min(params['max_position_size'] * 2.0, 5000.0)  # For backtesting only
            params['initial_risk_per_trade'] = min(params['initial_risk_per_trade'] * 1.2, 0.02)
            logger.info("Excellent performance - increasing position size and risk")
        elif profit_factor > 10.0 and win_rate > 0.65:
            # Good performance - be slightly more aggressive
            params['position_size_multiplier'] = min(params['position_size_multiplier'] * 1.2, 30.0)
            params['max_position_size'] = min(params['max_position_size'] * 1.5, 3000.0)  # For backtesting only
            params['initial_risk_per_trade'] = min(params['initial_risk_per_trade'] * 1.1, 0.015)
            logger.info("Good performance - slightly increasing position size and risk")
        elif profit_factor < 5.0 or win_rate < 0.5:
            # Poor performance - be more conservative
            params['position_size_multiplier'] = max(params['position_size_multiplier'] * 0.8, 5.0)
            params['max_position_size'] = max(params['max_position_size'] * 0.8, 50.0)
            params['initial_risk_per_trade'] = max(params['initial_risk_per_trade'] * 0.8, 0.005)
            logger.info("Poor performance - decreasing position size and risk")
        
        # Adjust stop loss and take profit based on average win/loss
        if avg_win_pct > 0.25:
            # Large average wins - can use wider stops and higher targets
            params['stop_loss_pct'] = min(params['stop_loss_pct'] * 1.2, 0.08)
            params['take_profit_pct'] = min(params['take_profit_pct'] * 1.2, 0.35)
            logger.info("Large average wins - widening stops and increasing targets")
        elif avg_loss_pct < -0.08:
            # Large average losses - need tighter stops
            params['stop_loss_pct'] = max(params['stop_loss_pct'] * 0.8, 0.03)
            logger.info("Large average losses - tightening stops")
        
        # Adjust signal generation based on trade frequency
        if total_trades < 20:
            # Too few trades - relax signal requirements
            params['confidence_threshold'] = max(params['confidence_threshold'] * 0.95, 0.75)
            params['force_signal_threshold'] = max(params['force_signal_threshold'] * 0.95, 0.70)
            params['min_signals_per_day'] += 1
            params['max_signals_per_day'] += 2
            logger.info("Too few trades - relaxing signal requirements")
        elif total_trades > 100:
            # Too many trades - tighten signal requirements
            params['confidence_threshold'] = min(params['confidence_threshold'] * 1.05, 0.95)
            params['force_signal_threshold'] = min(params['force_signal_threshold'] * 1.05, 0.90)
            params['min_signals_per_day'] = max(params['min_signals_per_day'] - 1, 1)
            params['max_signals_per_day'] = max(params['max_signals_per_day'] - 2, 3)
            logger.info("Too many trades - tightening signal requirements")
        
        # Adjust component weights based on performance
        if 'component_performance' in historical_performance:
            comp_perf = historical_performance['component_performance']
            
            # Normalize weights
            total_weight = (params['markov_confidence_weight'] + params['ml_confidence_weight'] + 
                          params['fractal_confidence_weight'] + params['price_action_weight'] + 
                          params['volume_confidence_weight'])
            
            # Adjust weights based on component performance
            if 'markov_win_rate' in comp_perf:
                params['markov_confidence_weight'] *= (comp_perf['markov_win_rate'] / win_rate)
            if 'ml_win_rate' in comp_perf:
                params['ml_confidence_weight'] *= (comp_perf['ml_win_rate'] / win_rate)
            if 'fractal_win_rate' in comp_perf:
                params['fractal_confidence_weight'] *= (comp_perf['fractal_win_rate'] / win_rate)
            if 'price_action_win_rate' in comp_perf:
                params['price_action_weight'] *= (comp_perf['price_action_win_rate'] / win_rate)
            if 'volume_win_rate' in comp_perf:
                params['volume_confidence_weight'] *= (comp_perf['volume_win_rate'] / win_rate)
            
            # Re-normalize weights
            new_total = (params['markov_confidence_weight'] + params['ml_confidence_weight'] + 
                        params['fractal_confidence_weight'] + params['price_action_weight'] + 
                        params['volume_confidence_weight'])
            
            normalization_factor = total_weight / new_total
            
            params['markov_confidence_weight'] *= normalization_factor
            params['ml_confidence_weight'] *= normalization_factor
            params['fractal_confidence_weight'] *= normalization_factor
            params['price_action_weight'] *= normalization_factor
            params['volume_confidence_weight'] *= normalization_factor
            
            logger.info("Adjusted component weights based on performance")
    
    # Ensure position size is capped for live trading safety
    # Note: For backtesting, we can use higher values, but in live trading this will be capped at 100.0
    params['max_position_size_live'] = min(params.get('max_position_size', 100.0), 100.0)
    
    # Log the generated parameters
    logger.info(f"Generated optimal parameters for {symbol} on {timeframe} timeframe")
    logger.info(f"Position Size Multiplier: {params['position_size_multiplier']}")
    logger.info(f"Max Position Size (Backtest): {params['max_position_size']}")
    logger.info(f"Max Position Size (Live): {params['max_position_size_live']}")
    logger.info(f"Stop Loss: {params['stop_loss_pct']*100:.2f}%")
    logger.info(f"Take Profit: {params['take_profit_pct']*100:.2f}%")
    
    return params
