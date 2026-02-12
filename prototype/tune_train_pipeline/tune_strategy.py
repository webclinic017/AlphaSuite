
import json
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import asdict

import numpy as np
import pybroker
from skopt import gp_minimize

from prototype.tune_train_pipeline.walkforward import WalkForward
from prototype.tune_train_pipeline.tune_train_base import ARTIFACTS_DIR
from pybroker_trainer.strategy_loader import load_strategy_class, get_strategy_tuning_space, get_strategy_defaults
from quant_engine import BASE_CONTEXT_COLUMNS, NumpyEncoder

logger = logging.getLogger(__name__)



"""
@cli.command(name='tune-strategy')
@click.option('--ticker', '-t', required=True, help='Stock ticker symbol.')
@click.option('--strategy-type', '-s', required=True, type=click.Choice(list(STRATEGY_CLASS_MAP.keys())), help='The type of strategy to tune.')
@click.option('--n-calls', default=100, help='Number of optimization iterations.')
@click.option('--start-date', default='2000-01-01', help='Start date for tuning data (YYYY-MM-DD).')
@click.option('--end-date', default=None, help='End date for tuning data (YYYY-MM-DD).')
@click.option('--commission', default=0.0, help='Commission cost per share (e.g., 0.005).')
@click.option('--max-drawdown', default=40.0, help='Maximum acceptable drawdown percentage for tuning objective.')
@click.option('--min-trades', default=20, help='Minimum acceptable trade count for tuning objective.')
@click.option('--min-win-rate', default=40.0, help='Minimum acceptable win rate for tuning objective.')
def tune_strategy(ticker, strategy_type, n_calls, start_date, end_date, commission, max_drawdown, min_trades, min_win_rate):
    '''Performs Bayesian optimization on strategy-level parameters.'''
    logger.info("running tune strategy...")
    tune_strategy(ticker.upper(), strategy_type, n_calls, start_date, end_date, commission, max_drawdown, min_trades, min_win_rate, progress_callback=None, stop_event_checker=None)
"""


def tune_strategy(portfolio, tickers, strategy_type, n_calls, start_date, end_date, commission, max_drawdown, min_trades, min_win_rate, progress_callback=None, stop_event_checker=None):
    """Core logic for Bayesian optimization on strategy-level parameters."""
    logger.info(f"--- Starting Strategy-Level Parameter Tuning for {tickers} with {strategy_type} ---")

    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    strategy_class = load_strategy_class(strategy_type)
    search_space = get_strategy_tuning_space(strategy_class) if strategy_class else []
    if not search_space:
        logger.error(f"Strategy tuning is not configured for strategy_type '{strategy_type}'.")
        return
    strategy_params = get_strategy_defaults(strategy_class)

    # --- Variables to store the metrics of the best run found during optimization ---
    best_run_metrics = {}
    best_score = float('inf')

    # Use a nonlocal counter for progress tracking instead of a global variable.
    iteration_count = 0

    _preloaded_data = False    # you can try either one
    #-----------------------------------------------------------------------------
    # get preloaded_data_df, preloaded_features
    if _preloaded_data :
        walkforward = WalkForward()
        preloaded_data_df = walkforward.init_run(
            tickers, strategy_type, False, None, None,
            start_date, end_date,
            False,
            False,
            True)
        preloaded_features = walkforward._features
    else:
        preloaded_data_df = None
        preloaded_features = None

    #-----------------------------------------------------------------------------
    def skopt_callback(res):
        """Callback for skopt to check for stop event."""
        if stop_event_checker and stop_event_checker():
            logger.warning("Stop event received. Halting optimization.")
            # Returning True stops the gp_minimize loop.
            return True
        return False

    # 2. Define the objective function to minimize
    def objective(params):
        nonlocal best_run_metrics, best_score, iteration_count # Allow modification of outer scope variables

        # --- NEW: Check for stop event at the beginning of each iteration ---
        # This makes the stop button feel instantaneous, as it will exit at the start
        # of the next iteration rather than waiting for the current long one to finish.
        if stop_event_checker and stop_event_checker():
            logger.warning("Stop event detected at start of objective function. Terminating this run.")
            return float \
                ('inf') # Return a high value to penalize this run. The main callback will then terminate the process.

        # --- Dynamically map the list of params from the optimizer to a dictionary ---

        # --- Update progress for UI ---
        iteration_count += 1
        if progress_callback:
            progress_callback(iteration_count / n_calls, f"Running iteration {iteration_count}/{n_calls}...")

        # This is more robust than hardcoding indices.
        param_dict = {dim.name: val for dim, val in zip(search_space, params)}

        # --- Prepare data with the specific parameters for this trial ---
        try:
            current_strategy_params = get_strategy_defaults(strategy_class)
            current_strategy_params.update(param_dict)

            strategy_instance = strategy_class(params=current_strategy_params)
            """
            # This is the crucial fix: run the full data prep on raw data for each trial.
            data_df = strategy_instance.prepare_data(data=base_data_df.copy())
            """
            features = strategy_instance.get_feature_list()
            context_columns_to_register = BASE_CONTEXT_COLUMNS + strategy_instance.get_extra_context_columns_to_register()

            is_ml_strategy = strategy_instance.is_ml_strategy
        except Exception as e:
            logger.error(f"Error during data preparation in objective function: {e}")
            return 1.0 # Return a poor score

        if is_ml_strategy:
            pybroker.register_columns(features)
        pybroker.register_columns(context_columns_to_register) # Always register context columns


        walkforward = WalkForward()
        result, quality_scores = walkforward.run_pybroker_walkforward(
            portfolio_name='tune_strategies',
            tickers=tickers,
            strategy_type=strategy_type,
            start_date=start_date, end_date=end_date, tune_hyperparameters=False,
            plot_results=False, save_assets=False,
            override_params=current_strategy_params, disable_inner_parallelism=True,
            commission_cost=commission,
            preloaded_data_df=preloaded_data_df,     #data_df,
            preloaded_features=preloaded_features,    #features,
            calc_bootstrap=False, # Disable bootstrapping to prevent parallel process errors with UI logging
            stop_event_checker=stop_event_checker # Pass the checker down
        )

        if result is None or result.metrics.trade_count < 5:
            logger.warning("Backtest failed or had < 5 trades. Assigning poor score.")
            return 1.0

        # --- Objective Function Calculation ---
        # This function defines what we want to optimize. It uses hard constraints
        # and a clear primary objective.
        metrics = result.metrics
        penalty = 1000.0

        # Check all hard constraints first. If any are violated, return a large penalty score.
        if metrics.max_drawdown > max_drawdown:
            logger.warning(f"Constraint VIOLATED: Max Drawdown {metrics.max_drawdown:.2f}% > {max_drawdown:.2f}%")
            return penalty + (metrics.max_drawdown - max_drawdown)
        if metrics.trade_count < min_trades:
            logger.warning(f"Constraint VIOLATED: Trade Count {metrics.trade_count} < {min_trades}")
            return penalty + (min_trades - metrics.trade_count)
        if metrics.win_rate < min_win_rate:
            logger.warning(f"Constraint VIOLATED: Win Rate {metrics.win_rate:.2f}% < {min_win_rate:.2f}%")
            return penalty + (min_win_rate - metrics.win_rate)

        # If all constraints are met, the objective is to maximize (Sortino * Profit Factor).
        # Since the optimizer minimizes, we return the negative of this product.
        objective_value = (metrics.sortino or 0) * (metrics.profit_factor or 0)
        score = -objective_value

        logger.info \
            (f"--> Params: {param_dict} | Sortino: {metrics.sortino:.4f} | Profit Factor: {metrics.profit_factor:.4f} | Score: {score:.4f}")

        # Add penalty for models with low predictive power (AUC), guiding the optimizer toward more reliable strategies
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0

        # --- FIX: Only apply model quality penalty to ML strategies ---
        if is_ml_strategy and avg_quality < 0.52:
            # If average model quality is below random chance, penalize heavily.
            quality_penalty = (0.52 - avg_quality) * 10  # Penalty increases as quality decreases.
            score += quality_penalty
            logger.info \
                (f"--> POOR MODEL QUALITY (Avg AUC: {avg_quality:.3f}). Applying penalty: {quality_penalty:.4f}. Final Score: {score:.4f}")

        # --- If this is the best score so far, save its metrics for logging later ---
        if score < best_score:
            best_score = score
            best_run_metrics = asdict(result.metrics)

        return score

    # 3. Run the Bayesian optimization
    opt_result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls, random_state=42, n_jobs=1 if progress_callback else -1,
        callback=skopt_callback if stop_event_checker else None)

    # Check if the process was stopped prematurely
    if stop_event_checker and stop_event_checker():
        logger.info("Tuning process was stopped by the user. Aborting without saving results.")
        return

    # 4. Print the best results
    best_params_dict = {dim.name: val for dim, val in zip(search_space, opt_result.x)}
    logger.info("\n--- Strategy Tuning Complete ---")
    logger.info(f"Best parameters found: ")
    for name, value in best_params_dict.items():
        logger.info(f"  - {name}: {value:.4f}" if isinstance(value, float) else f"  - {name}: {value}")
    logger.info(f"Best score achieved during tuning: {opt_result.fun:.4f}")

    # --- Log the full metrics that corresponded to the best score ---
    if best_run_metrics:
        logger.info("--- Metrics for the Best Run Found During Tuning (Strategy Rules + Default Model) ---")
        # Pretty print the dictionary
        for key, value in best_run_metrics.items():
            if isinstance(value, (float, Decimal)):
                logger.info(f"  {key}: {float(value):.4f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("------------------------------------------------------------------------------------")

    # --- Save the best parameters to a file ---
    model_dir = ARTIFACTS_DIR
    os.makedirs(model_dir, exist_ok=True)
    tuned_params_filename = os.path.join(model_dir, f'{portfolio}_{strategy_type}_best_strategy_params.json')
    with open(tuned_params_filename, 'w') as f:
        # --- Add the versioning flag to the best params file ---
        params_to_save = best_params_dict.copy()
        params_to_save['ratios_annualized'] = True
        json.dump(params_to_save, f, indent=4, cls=NumpyEncoder)
    logger.info(f"Saved best strategy parameters to {tuned_params_filename}")

    # --- Re-run the final backtest with the best parameters to save the results ---
    # This ensures that visualize-model will show the performance of the tuned strategy.
    logger.info("\n--- Re-running final backtest with best parameters to save artifacts... ---")
    # For ML strategies, we also tune the model's hyperparameters on this final run.
    # For rule-based strategies, the `tune_hyperparameters` flag has no effect.
    walkforward = WalkForward()

    result, quality_scores = walkforward.run_pybroker_walkforward(
        portfolio_name='tune_strategies',
        tickers=tickers,
        strategy_type=strategy_type,
        start_date=start_date,
        end_date=end_date,
        tune_hyperparameters=True,
        plot_results=False,
        save_assets=True,
        override_params=best_params_dict,
        commission_cost=commission,
        use_tuned_strategy_params=False # We are overriding directly
    )
    logger.info(f"--- Artifacts for best parameters saved. You can now use 'visualize-model'. ---")
    return result, quality_scores

