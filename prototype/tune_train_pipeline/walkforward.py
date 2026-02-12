import json
import logging
import os
import traceback
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybroker
from prototype.tune_train_pipeline.tune_train_base import log_walkforward_split_dates, save_walkforward_artifacts, \
    load_hyper_params, ARTIFACTS_DIR
from lightgbm import LGBMClassifier
from pybroker import PositionMode, StrategyConfig
from pybroker_trainer.config_loader import load_strategy_config
from pybroker_trainer.strategy_loader import load_strategy_class, get_strategy_defaults
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from skopt import gp_minimize
from skopt.space import Real, Integer

from quant_engine import BASE_CONTEXT_COLUMNS, _prepare_base_data, PassThroughModel, custom_predict_fn, \
    ExpandingWindowStrategy, prepare_metrics_df_for_display, plot_performance_vs_benchmark, plot_feature_importance
from tools.file_wrapper import convert_to_json_serializable

# --- Logging Configuration ---
logger = logging.getLogger(__name__)

pd.set_option('future.no_silent_downcasting', True)

class WalkForward:

    def __init__(self, collect_train_data=False):
        self.reset()
        self._collect_train_data = collect_train_data

    def reset(self):
        self._tickers = None
        self._strategy_type = None
        self._features = []
        self._context_columns_to_register = []
        self._last_best_params = None
        self._last_trained_model = None
        self._all_feature_importances = []
        self._is_ml = True
        self._all_quality_scores = []
        self._strategy_instance = None
        self._strategy_params = None
        self._tune_hyperparameters = False
        self._stop_event_checker = None
        self._start_date = None
        self._end_date = None
        self._model = None
        self._hyper_params = None
        self._total_train_data = None
        self._portfolio_name = None

    def get_model(self, **kwargs):
        if self._model is None:
            self._model = LGBMClassifier(verbosity=-1, **kwargs)
        return self._model

    def append_train_data(self, train_data):
        if not self._collect_train_data:
            return
        if self._total_train_data is None:
            self._total_train_data = train_data
        else:
            self._total_train_data = pd.concat([self._total_train_data, train_data], ignore_index=True)

    def init_run(self, portfolio_name, tickers:list[str], strategy_type, tune_hyperparameters,  preloaded_data_df, preloaded_features, start_date, end_date, use_tuned_strategy_params, override_params, disable_inner_parallelism, use_tuned_hyper_params):
        self.reset()
        self._portfolio_name = portfolio_name
        self._tickers = tickers
        self._start_date = start_date
        if end_date is None:
            self._end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            self._end_date = end_date

        self._strategy_type = strategy_type
        self._tune_hyperparameters = tune_hyperparameters

        # --- Load strategy config from JSON file ---
        self._strategy_class = load_strategy_class(self._strategy_type)
        base_params = get_strategy_defaults(self._strategy_class) if self._strategy_class else {}
        base_params.update({'disable_inner_parallelism': disable_inner_parallelism, })

        # Load tuned strategy parameters if requested
        if use_tuned_strategy_params:
            self._strategy_params = load_strategy_config(self._strategy_type, base_params)
        else:
            self._strategy_params = base_params

        # --------------------------------------------------------------
        if use_tuned_hyper_params:
            self._hyper_params = load_hyper_params(self._strategy_type)
        else:
            self._hyper_params = dict()

        # --------------------------------------------------------------
        # Allow overriding parameters for optimization
        if override_params:
            logger.info(f"Overriding strategy parameters with: {override_params}")
            self._strategy_params.update(override_params)

        if self._strategy_class:
            self._strategy_instance = self._strategy_class(params=self._strategy_params)
            self._is_ml = self._strategy_instance.is_ml_strategy
        else:
            self._strategy_instance = None
            self._is_ml = False

        if preloaded_data_df is not None and preloaded_features is not None:
            self._features = preloaded_features
        else:
            if self._strategy_class:
                self._features = self._strategy_instance.get_feature_list()
                self._context_columns_to_register = BASE_CONTEXT_COLUMNS + self._strategy_instance.get_extra_context_columns_to_register()
            else:  # Fallback for legacy strategies
                raise NotImplementedError(
                    f"Strategy '{self._strategy_type}' has not been migrated to the new encapsulated format. Please create a strategy module for it.")
        final_df = pd.DataFrame()
        for ticker in tickers:
            df = self.init_ticker(ticker, preloaded_data_df, preloaded_features)
            final_df = pd.concat([final_df, df], ignore_index=True)
        return final_df


    def init_ticker(self, ticker, preloaded_data_df, preloaded_features):
        # --- OPTIMIZATION: Use pre-loaded data if provided (e.g., from tune_strategy) ---
        if preloaded_data_df is not None and preloaded_features is not None:
            logger.info("Using pre-loaded data for backtest.")
            data_df = preloaded_data_df
        else:
            if self._strategy_class:
                base_df = _prepare_base_data(ticker, self._start_date, self._end_date, self._strategy_params)
                data_df = self._strategy_instance.prepare_data(data=base_df)
            else:  # Fallback for legacy strategies
                raise NotImplementedError(
                    f"Strategy '{self._strategy_type}' has not been migrated to the new encapsulated format. Please create a strategy module for it.")
        return data_df


    def check_min_total_setups_for_run(self, data_df, min_total_setups_for_run):
        if self._is_ml:
            # --- NEW: Add a global check for minimum setups before starting the walk-forward ---
            # min_total_setups_for_run = 60  # A reasonable minimum for the entire dataset
            valid_setups = data_df.dropna(subset=['target'])
            if len(valid_setups) < min_total_setups_for_run:
                logger.error(
                    f"Strategy '{self._strategy_type}' has only {len(valid_setups)} total setups.")
                logger.error(
                    f"This is insufficient for a reliable walk-forward backtest (min: {min_total_setups_for_run}).")
                logger.error(
                    "Consider using the 'pre-scan-universe' command to find tickers with more frequent setups.")
                return False  # Abort the run early
        return True

    def train_fn_check_min_requirement(self, symbol, train_data, **model_config):
        # --- More robust check for minimum samples per fold ---
        min_total_samples = 30  # Increased minimum total setups required for training
        min_class_samples = 10  # Minimum required setups for EACH class (win and loss)

        if train_data.empty or len(train_data) < min_total_samples:
            logger.warning(
                f"[{symbol}] Training data is empty or has insufficient samples ({len(train_data)} < {min_total_samples}). No model will be trained for this fold.")
            model = LGBMClassifier(verbosity=-1, random_state=42, n_jobs=1, **model_config)
            return {'model': model, 'features': self._features}

        # Check for minimum samples in the minority class
        if 'target' in train_data.columns and not train_data['target'].value_counts().empty:
            if len(train_data['target'].unique()) < 2 or train_data[
                'target'].value_counts().min() < min_class_samples:
                logger.warning(
                    f"[{symbol}] Minority class has too few samples ({train_data['target'].value_counts().min()} < {min_class_samples}). No model will be trained for this fold.")
                model = LGBMClassifier(verbosity=-1, random_state=42, n_jobs=1, **model_config)
                return {'model': model, 'features': self._features}
        return None

    def train_fn_prepare_train_data(self, train_data, symbol):
        # --- Filter training data to only include rows with valid targets ---
        if 'target' in train_data.columns:
            train_data = train_data.dropna(subset=['target'])

        # --- Apply embargo to prevent data leakage from future information ---
        # The embargo period is based on the `target_eval_bars` or `stop_out_window`
        # parameter, which defines how many bars into the future the target is calculated.
        # We remove data points from the end of the training set that would overlap
        # with the target calculation window of the test set.
        # This is crucial for preventing look-ahead bias in walk-forward validation.
        if not train_data.empty:
            eval_bars = self._strategy_instance.params.get('target_eval_bars')
            if eval_bars is None:
                eval_bars = self._strategy_instance.params.get('stop_out_window',
                                                               15)  # Default to 15 if neither is found
            last_train_date = train_data['date'].max()
            embargo_start_date = last_train_date - pd.Timedelta(days=eval_bars * 1.5)  # Use 1.5 for a safety margin
            original_len = len(train_data)
            train_data = train_data[train_data['date'] < embargo_start_date]
            logger.info(
                f"[{symbol}] Applied embargo: Purged {original_len - len(train_data)} samples from the end of the training set.")
        return train_data

    # -------------------------------------------------------------------------
    # Define the training function for the model.
    # This function will be called by PyBroker for each walk-forward window.
    def train_fn(self, symbol, train_data, test_data, **kwargs):
        from sklearn.model_selection import train_test_split  # Local import for this function
        model_config = self._strategy_instance.get_model_config()
        # --- NEW: Enhanced logging and checks for data validity ---
        train_start = train_data['date'].min().date() if not train_data.empty else 'N/A'
        train_end = train_data['date'].max().date() if not train_data.empty else 'N/A'
        logger.info(f"[{symbol}] Training fold: {train_start} to {train_end}. Initial samples: {len(train_data)}")

        pybroker.disable_logging()

        if train_data.empty:
            logger.warning(
                f"[{symbol}] Training data is empty for this fold. This is expected if the stock did not exist for the full period. Returning untrained model.")
            model = LGBMClassifier(verbosity=-1, random_state=42, n_jobs=1, **model_config)
            return {'model': model, 'features': self._features}

        train_data = self.train_fn_prepare_train_data(train_data, symbol)
        self.append_train_data(train_data)
        model_bundle = self.train_fn_check_min_requirement(symbol, train_data, **model_config)
        if model_bundle is not None: return model_bundle    # failed of min_requirement check

        # --- Determine if tuning is feasible for this specific fold ---
        can_tune = self._tune_hyperparameters
        if can_tune:
            n_splits = 3  # Must match the cv_splitter below
            min_samples_for_tuning = 20  # A reasonable minimum to attempt tuning
            # Check if any class has fewer samples than n_splits, which would break StratifiedKFold
            if 'target' in train_data.columns and not train_data['target'].value_counts().empty:
                min_class_count = train_data['target'].value_counts().min()
                if min_class_count < n_splits or len(train_data) < min_samples_for_tuning:
                    logger.warning(
                        f"[{symbol}] Insufficient samples for tuning (Total: {len(train_data)}, Min Class: {min_class_count}).")
                    logger.warning(
                        f"[{symbol}] Disabling hyperparameter tuning for this fold and using default parameters.")
                    can_tune = False
            else:
                logger.warning(
                    f"[{symbol}] Target column missing or empty in train_data. Disabling tuning for this fold.")
                can_tune = False

        if can_tune:
            best_params = self.tune_hyperparameters_with_gp_minimize(
                train_data=train_data,
                features=self._features,
                model_config=model_config,
                stop_event_checker=self._stop_event_checker,
                symbol=symbol
            )

            final_model = self.get_model(random_state=42, n_jobs=-1, class_weight='balanced', **model_config,
                                         **best_params)
            final_model.fit(train_data[self._features], train_data['target'].astype(int))

            self._last_best_params = best_params
            self._last_trained_model = final_model
            if hasattr(final_model, 'feature_importances_'):
                self._all_feature_importances.append(final_model.feature_importances_)
            return {'model': final_model, 'features': self._features}
        else:
            # --- Train with default hyperparameters (no tuning) ---
            logger.info("Running with default hyperparameters (no tuning).")
            # --- Train LGBM with default hyperparameters ---
            try:
                sub_train, sub_val = train_test_split(train_data, test_size=0.2, random_state=42,
                                                      stratify=train_data['target'])
            except ValueError:
                logger.warning(
                    f"[{symbol}] Could not create validation split due to class imbalance. Proceeding without quality gate for this fold.")
                sub_train, sub_val = train_data, pd.DataFrame()

            n_jobs = 1 if self._strategy_params.get('disable_inner_parallelism') else -1
            default_lgbm_params = {'random_state': 42, 'n_jobs': n_jobs, 'class_weight': 'balanced',
                                   'min_child_samples': 5, **model_config, **self._hyper_params}

            temp_model = self.get_model(**default_lgbm_params)
            temp_model.fit(sub_train[self._features], sub_train['target'].astype(int))

            # Add a model quality gate to prevent poorly performing models from being used in backtest
            auc_score = 0.5
            if not sub_val.empty and len(np.unique(sub_val['target'])) > 1:
                val_preds = temp_model.predict_proba(sub_val[self._features])
                if model_config.get('objective') == 'binary':
                    auc_score = roc_auc_score(sub_val['target'], val_preds[:, 1])
                else:
                    auc_score = roc_auc_score(sub_val['target'], val_preds, multi_class='ovr')

            # Track the quality score for this fold
            self._all_quality_scores.append(auc_score)

            min_auc_threshold = 0.52
            if auc_score < min_auc_threshold:
                logger.warning(
                    f"[{symbol}] Model quality check failed for this fold (AUC: {auc_score:.3f} < {min_auc_threshold}).")
                logger.warning(f"[{symbol}] Discarding trained model and using a pass-through model instead.")
                final_model = PassThroughModel(n_classes=model_config.get('num_class', 2))
            else:
                logger.info(
                    f"[{symbol}] Model quality check passed (AUC: {auc_score:.3f}). Retraining on full fold data.")
                final_model = self.get_model(**default_lgbm_params)
                final_model.fit(train_data[self._features], train_data['target'].astype(int))
                if hasattr(final_model, 'feature_importances_'):
                    self._all_feature_importances.append(final_model.feature_importances_)

            self._last_trained_model = final_model
            return {'model': final_model, 'features': self._features}
            # ------- end of train_fn() ------------------------------------------------------

    def run_pybroker_walkforward(self, portfolio_name: str,  tickers: list[str],
                                 start_date: str = '2000-01-01', end_date: Optional[str] = None,
                                 strategy_type: str = 'trend_following', tune_hyperparameters: bool = True,
                                 plot_results: bool = True, save_assets: bool = False, override_params: dict = None,
                                 use_tuned_strategy_params: bool = False, disable_inner_parallelism: bool = False,
                                 preloaded_data_df: pd.DataFrame = None, preloaded_features: list = None,
                                 commission_cost: float = 0.0, calc_bootstrap: bool = True, stop_event_checker=None,
                                 use_tuned_hyper_params: bool = False
                                 ):
        """
        Runs the full walk-forward analysis for a given ticker.
        """
        try:
            # --- Unify default end_date handling ---
            data_df = self.init_run(portfolio_name, tickers, strategy_type, tune_hyperparameters, preloaded_data_df, preloaded_features, start_date, end_date, use_tuned_strategy_params, override_params, disable_inner_parallelism, use_tuned_hyper_params)

            if not self.check_min_total_setups_for_run(data_df, 60):
                return None, []
            # Filter out features that might not have been calculated or don't exist
            # _features = [f for f in data_df.columns if f in _features]
            # data_df.dropna(subset=_features + ['target'], inplace=True)
            if self._is_ml:
                pybroker.register_columns(self._features)
            pybroker.register_columns(self._context_columns_to_register)  # Always register context columns
            def model_input_data_fn(data):
                return data[self._features]

            # Step 3: Register the model with PyBroker.
            # We don't pass `indicators` because they are already calculated and part of `data_df`.
            model_name = 'binary_classifier'
            model_source = pybroker.model(name=model_name, fn=self.train_fn, predict_fn=custom_predict_fn,
                                          input_data_fn=model_input_data_fn)

            # Step 4: Configure StrategyConfig and instantiate the correct trader
            strategy_config = StrategyConfig(
                position_mode=PositionMode.LONG_ONLY,
                fee_mode=pybroker.FeeMode.PER_SHARE if commission_cost > 0 else None,
                fee_amount=commission_cost
            )
            strategy = ExpandingWindowStrategy(data_source=data_df, start_date=self._start_date, end_date=self._end_date,
                                               config=strategy_config)
            ticker_params = dict.fromkeys(tickers, self._strategy_params)
            trader = self._strategy_instance.get_trader(model_name if self._is_ml else None,ticker_params)

            if self._is_ml:
                models_to_use = model_source if isinstance(model_source, list) else [model_source]
                strategy.add_execution(trader.execute, tickers, models=models_to_use)
            else:
                # Non-ML strategies don't need a model passed to their execution
                strategy.add_execution(trader.execute, tickers)

            # Step 5: Run the walk-forward analysis
            self._all_feature_importances = []  # Reset before running
            logger.info("Starting PyBroker walk-forward analysis...")
            total_years = (data_df['date'].max() - data_df['date'].min()).days / 365.25 if not data_df.empty else 0
            if total_years < 4:  # Need at least ~4 years for a meaningful split
                logger.error(f"Not enough data ({total_years:.2f} years) for a walk-forward. Minimum 4 years required.")
                return None
            windows = (total_years >= 20) + (total_years >= 15) + (total_years >= 8) + 1
            # `train_size` is not used for expanding windows but is a required parameter.
            train_size_prop = 0.7
            logger.info(f"Total data history: {total_years:.2f} years. Using {windows} walk-forward windows.")
            # --- Log the walk-forward split dates for clarity ---
            log_walkforward_split_dates(data_df, self._start_date, self._end_date, strategy_config, windows, train_size_prop)

            result = strategy.walkforward(
                windows=windows,
                train_size=train_size_prop,
                lookahead=1,
                calc_bootstrap=calc_bootstrap,
                warmup=2,
            )

            # --- Annualize Sharpe and Sortino Ratios ---
            if result and hasattr(result, 'metrics') and hasattr(result, 'metrics_df'):
                # The result object is immutable. We create a reset object with the annualized metrics_df.
                display_metrics_df = prepare_metrics_df_for_display(result.metrics_df, '1d')
                savable_result = replace(result, metrics_df=display_metrics_df)
            else:
                savable_result = result  # Fallback if result is None or malformed
            if save_assets:
                save_walkforward_artifacts(self, portfolio_name, result=savable_result)  # Save the object with the annualized metrics

            logger.info(f"\n--- Walk-Forward Analysis Results for {tickers} with {self._strategy_type} strategy ---")
            logger.info(
                "NOTE: These metrics are calculated on out-of-sample test windows only, providing a more realistic performance estimate.")
            logger.info(
                display_metrics_df.to_string() if 'display_metrics_df' in locals() else result.metrics_df.to_string())
            if plot_results:
                fig_equity = plot_performance_vs_benchmark(result, f'Walk-Forward Equity Curve for {tickers}')
                if fig_equity:
                    plt.show()
                if self._all_feature_importances:
                    fig_importance = plot_feature_importance(self._features, self._all_feature_importances)
                    if fig_importance:
                        plt.show()

        except Exception as e:
            logger.error(f"An error occurred during the PyBroker walk-forward analysis for {tickers}: {e}")
            traceback.print_exc()
            return None, []
        finally:
            # Unregister columns to clean up global scope
            if self._features:
                pybroker.unregister_columns(self._features)
            if self._context_columns_to_register:
                pybroker.unregister_columns(self._context_columns_to_register)
        return result, self._all_quality_scores  # Return the result object and quality scores
        # ------end of run_pybroker_walkforward ---------------------------------------------------

    def tune_save_hyperparameters(self):
        if self._total_train_data is None:
            logger.info("No train_data to tune!")
            return
        self._last_best_params = self.tune_hyperparameters_with_gp_minimize(
            train_data=self._total_train_data,
            features=self._features,
            model_config=self._strategy_instance.get_model_config(),
            stop_event_checker=None,
            symbol="total"
        )
        logger.info(f" Best hyperparameters found: {self._last_best_params}")

        params_filename = os.path.join(ARTIFACTS_DIR, f'{self._portfolio_name}_{self._strategy_type}_best_params.json')
        with open(params_filename, 'w') as f:
            json.dump(convert_to_json_serializable(self._last_best_params), f, indent=4)
        logger.info(f"Saved best hyperparameters from last fold to {params_filename}")


    @staticmethod
    def tune_hyperparameters_with_gp_minimize(train_data, features, model_config, stop_event_checker, symbol):
        """
        Performs hyperparameter tuning for the LGBMClassifier using gp_minimize.
        This offers fine-grained control and supports an interruptible callback.
        Returns a dictionary of the best parameters found.
        """
        from sklearn.model_selection import cross_val_score  # Local import
        logger.info(f"[{symbol}] Starting hyperparameter tuning with gp_minimize...")

        search_spaces_dict = {
            'learning_rate': Real(0.01, 0.3, 'log-uniform'),
            'n_estimators': Integer(50, 500),
            'num_leaves': Integer(20, 100),
            'max_depth': Integer(-1, 50),
            'reg_alpha': Real(0.0, 1.0, 'uniform'),  # L1 regularization
            'reg_lambda': Real(0.0, 1.0, 'uniform'),  # L2 regularization
        }
        dimensions = list(search_spaces_dict.values())
        dimensions_names = list(search_spaces_dict.keys())

        def hp_objective(params):
            """Objective function for gp_minimize to maximize cross-validated AUC."""
            if stop_event_checker and stop_event_checker():
                logger.warning(f"[{symbol}] Stop event detected in HP tuning objective. Aborting this run.")
                return 1.0  # Return a high value (bad score) for minimization

            param_dict = {name: val for name, val in zip(dimensions_names, params)}
            model = LGBMClassifier(verbosity=-1, random_state=42, n_jobs=1, class_weight='balanced', **model_config, **param_dict)

            cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scoring_metric = 'roc_auc_ovr' if model_config.get('objective') == 'multiclass' else 'roc_auc'

            scores = cross_val_score(model, train_data[features], train_data['target'].astype(int), cv=cv_splitter,
                                     scoring=scoring_metric, n_jobs=-1)

            mean_score = np.mean(scores)
            # We want to maximize the score, so we return its negative for minimization
            return -mean_score

        def skopt_callback(res):
            """Callback for gp_minimize to check for stop event."""
            if stop_event_checker and stop_event_checker():
                logger.warning(f"[{symbol}] Stop event received during hyperparameter tuning. Halting search.")
                return True  # Returning True stops the optimization loop.
            return False

        opt_result = gp_minimize(
            func=hp_objective,
            dimensions=dimensions,
            n_calls=32,
            random_state=42,
            callback=skopt_callback
        )

        best_params = {name: val for name, val in zip(dimensions_names, opt_result.x)}
        logger.info(f"[{symbol}] Best hyperparameters found: {best_params}")
        return best_params


