import logging
import traceback
from typing import Optional, Dict
import pandas as pd
import pybroker
from lightgbm import LGBMClassifier
from pybroker import StrategyConfig, PositionMode, Strategy
from prototype.tune_train_pipeline.tune_train_base import load_model
from pybroker_trainer.config_loader import load_strategy_config
from pybroker_trainer.strategy_loader import load_strategy_class, get_strategy_defaults
from quant_engine import _prepare_base_data, BASE_CONTEXT_COLUMNS, custom_predict_fn, \
    prepare_metrics_df_for_display, ExpandingWindowStrategy, plot_performance_vs_benchmark, PassThroughModel
from enum import Enum

logger = logging.getLogger(__name__)


class ModelMode(Enum):
    TRAINED = "useTrainedModel"
    PASSTHROUGH = "usePassThrough"
    DEFAULT = "default"


def run_pybroker_portfolio_backtest(portfolio_name: str, tickers: list[str], strategy_type: str,
                                    start_date: str, end_date: str,
                                    plot_results: bool = True,
                                    use_tuned_strategy_params: bool = False,
                                    model_mode: ModelMode = ModelMode.DEFAULT,
                                    max_open_positions: int = 5, commission_cost: float = 0.0):
    """
    Runs a single walk-forward backtest on a portfolio of tickers.
    This is used to validate scanning-based strategies like RSI Divergence.
    Extensions :
        use_tuned_strategy_params, use_trained_model to validate strategy_params, trained_model
        ./WORK/strategy_configs/{strategy_type}.json
        ./WORK/strategy_configs/{strategy_type}_model.pkl
    """
    all_data_dfs = []
    features = []
    context_columns_to_register = []
    params_map: Dict[str, dict] = {}

    try:
        logger.info(f"--- Preparing data for portfolio backtest across {len(tickers)} tickers... ---")
        strategy_class = load_strategy_class(strategy_type)
        if not strategy_class:
            logger.error(f"Could not load strategy class for {strategy_type}. Aborting.")
            return None
        is_ml = strategy_class(params={}).is_ml_strategy

        for ticker in tickers:
            try:
                # --- Load tuned parameters for each ticker if requested ---
                current_strategy_params = get_strategy_defaults(strategy_class) if strategy_class else {}

                if use_tuned_strategy_params:
                    # --- ./WORK/strategy_configs/{strategy_type}.json
                    current_strategy_params = load_strategy_config(strategy_type, current_strategy_params)
                    logger.info(f"Loaded tuned strategy parameters for {ticker}.")

                params_map[ticker] = current_strategy_params
                strategy_instance = strategy_class(params=current_strategy_params)
                base_df = _prepare_base_data(ticker, start_date, end_date, current_strategy_params)
                data_df = strategy_instance.prepare_data(data=base_df)
                if not data_df.empty:
                    all_data_dfs.append(data_df)
                    if not features:  # Capture feature list from the first successful ticker
                        features = strategy_instance.get_feature_list()
                else:
                    logger.warning(f"No data for {ticker}, skipping.")
            except Exception as e:
                logger.error(f"Failed to prepare data for {ticker}: {e}")

        if not all_data_dfs:
            logger.error("No data could be prepared for any tickers. Aborting portfolio backtest.")
            return None

        portfolio_df = pd.concat(all_data_dfs, ignore_index=True)
        logger.info(f"Combined portfolio data shape: {portfolio_df.shape}")

        # --- Setup PyBroker for Portfolio Backtest ---
        pybroker.register_columns(features)
        context_columns_to_register = BASE_CONTEXT_COLUMNS + strategy_instance.get_extra_context_columns_to_register()
        pybroker.register_columns(context_columns_to_register)

        def model_input_data_fn(data):
            return data[features]

        # The train_fn will be called by pybroker for each symbol in the portfolio.
        # We can reuse the same logic as the single-ticker backtest.
        def train_fn(symbol, train_data, test_data, **kwargs):
            # This function is a placeholder for ML strategies.
            # For non-ML, it won't be used, but pybroker requires it to be defined.
            train_start = train_data['date'].min().date() if not train_data.empty else 'N/A'
            train_end = train_data['date'].max().date() if not train_data.empty else 'N/A'
            logger.info(f"[{symbol}] Training fold: {train_start} to {train_end}. Initial samples: {len(train_data)}")

            pybroker.disable_logging()
            if train_data.empty:
                logger.warning(
                    f"[{symbol}] Training data is empty for this fold. This is expected if the stock did not exist for the full period. Returning untrained model.")
                model_config = strategy_instance.get_model_config()
                model = LGBMClassifier(verbosity=-1, random_state=42, n_jobs=1, **model_config)
                return {'model': model, 'features': features}

            # --- FIX: Drop rows where the target is NaN before training ---
            # This is crucial for strategies like RSI Divergence that only label specific setup days.
            if 'target' in train_data.columns:
                train_data = train_data.dropna(subset=['target'])

            if model_mode == ModelMode.PASSTHROUGH :
                return {'model': PassThroughModel(), 'features': features}

            # -- use trained model
            if  model_mode == ModelMode.TRAINED :
                # --- ./WORK/strategy_configs/{strategy_type}_model.pkl
                model = load_model(strategy_type)
                if model :
                    logger.info(f"Loaded trained model for {symbol}.")
                    return {'model': model, 'features': features}

            # --- NEW: More robust check for minimum samples per fold ---
            min_total_samples = 30  # Increased minimum total setups required for training
            min_class_samples = 10  # Minimum required setups for EACH class (win and loss)
            if train_data.empty or len(train_data) < min_total_samples:
                logger.warning(
                    f"[{symbol}] Training data is empty or has insufficient samples ({len(train_data)} < {min_total_samples}) for portfolio fold. Returning untrained model.")
                model_config = strategy_instance.get_model_config()
                model = LGBMClassifier(verbosity=-1, random_state=42, n_jobs=1, **model_config)
                return {'model': model, 'features': features}

            if 'target' in train_data.columns and not train_data['target'].value_counts().empty:
                if len(train_data['target'].unique()) < 2 or train_data[
                    'target'].value_counts().min() < min_class_samples:
                    logger.warning(
                        f"[{symbol}] Minority class has too few samples ({train_data['target'].value_counts().min()} < {min_class_samples}) for portfolio fold. Returning untrained model.")
                    model_config = strategy_instance.get_model_config()
                    model = LGBMClassifier(verbosity=-1, random_state=42, n_jobs=1, **model_config)
                    return {'model': model, 'features': features}

            logger.info(f"[{symbol}] Training model on {len(train_data)} valid setup samples...")
            model = LGBMClassifier(verbosity=-1, random_state=42, n_jobs=-1, class_weight='balanced',
                                   **strategy_instance.get_model_config())
            model.fit(train_data[features], train_data['target'].astype(int))
            return {'model': model, 'features': features}

        model_name = f"{strategy_type}_portfolio_model"
        model_source = pybroker.model(name=model_name, fn=train_fn, predict_fn=custom_predict_fn,
                                      input_data_fn=model_input_data_fn)

        strategy_config = StrategyConfig(
            position_mode=PositionMode.LONG_ONLY,
            exit_on_last_bar=True,
            max_long_positions=max_open_positions,
            fee_mode=pybroker.FeeMode.PER_SHARE if commission_cost > 0 else None,
            fee_amount=commission_cost
        )
        strategy = ExpandingWindowStrategy(data_source=portfolio_df, start_date=start_date, end_date=end_date,
                                           config=strategy_config)

        trader = strategy_instance.get_trader(model_name if is_ml else None, params_map)
        # Add a single execution for all tickers. PyBroker will handle the rest.
        if is_ml:
            strategy.add_execution(trader.execute, tickers, models=[model_source])
        else:
            strategy.add_execution(trader.execute, tickers)

        # --- Run the Walk-Forward Analysis ---
        logger.info("Starting portfolio walk-forward analysis...")
        total_years = (portfolio_df['date'].max() - portfolio_df['date'].min()).days / 365.25
        windows = 4 if total_years >= 20 else 2 if total_years >= 10 else 1

        result = strategy.walkforward(windows=windows, train_size=0.7, lookahead=1, calc_bootstrap=True)

        # --- Annualize Sharpe and Sortino Ratios ---
        if result and hasattr(result, 'metrics') and hasattr(result, 'metrics_df'):
            # The result object is immutable. We create a separate, annualized version for display/logging.
            display_metrics_df = prepare_metrics_df_for_display(result.metrics_df, '1d')

        logger.info(f"\n--- Portfolio Walk-Forward Results for {strategy_type} strategy ---")
        logger.info(
            display_metrics_df.to_string() if 'display_metrics_df' in locals() else result.metrics_df.to_string())
        if plot_results:
            plot_performance_vs_benchmark(result, f"Portfolio Performance ({strategy_type})")

    except Exception as e:
        logger.error(f"An error occurred during the portfolio backtest: {e}")
        traceback.print_exc()
        return None
    finally:
        if features: pybroker.unregister_columns(features)
        if context_columns_to_register: pybroker.unregister_columns(context_columns_to_register)
    return result
