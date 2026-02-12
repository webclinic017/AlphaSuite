from datetime import datetime
import pandas as pd
from prototype.tune_train_pipeline.tune_strategy import tune_strategy
from prototype.tune_train_pipeline.tune_train_base import copy_to_strategy_configs, print_full, is_ml_strategy
from prototype.tune_train_pipeline.walkforward import WalkForward
from prototype.tune_train_pipeline.backtest_portfolio import run_pybroker_portfolio_backtest, ModelMode


def run_tune_strategy(portfolio_name, tickers, strategy_type, start_date, end_date):
    start = datetime.now()
    result, all_quality_scores = tune_strategy(
        portfolio=portfolio_name,  # prefix of result files
        tickers=tickers,
        strategy_type=strategy_type,
        n_calls=150,
        start_date=start_date,
        end_date=end_date,
        commission=0.0,
        max_drawdown=40.0,
        min_trades=80.0,
        min_win_rate=35.0,

    )
    print(f"total duration: {datetime.now() - start}")
    print(f"all_quality_scores={all_quality_scores}")
    print_full(result.metrics_df)
    return result

def run_tune_hyper_params(portfolio_name, tickers, strategy_type, start_date, end_date):
    walkforward = WalkForward(collect_train_data=True)

    result, all_quality_scores = walkforward.run_pybroker_walkforward(
        portfolio_name=portfolio_name, # prefix of result files
        tickers=tickers,
        strategy_type= strategy_type,
        start_date=start_date,
        end_date=end_date,
        tune_hyperparameters=False,
        plot_results=False,
        save_assets=False,
        use_tuned_strategy_params=True
    )

    walkforward.tune_save_hyperparameters()

    print(f"all_quality_scores={all_quality_scores}")
    print_full(result.metrics_df)
    return result

def run_train_model(portfolio_name, tickers, strategy_type, start_date, end_date):
    start = datetime.now()
    walkforward = WalkForward()

    result, all_quality_scores = walkforward.run_pybroker_walkforward(
        portfolio_name=portfolio_name, # prefix of result files
        tickers=tickers,
        strategy_type= strategy_type,
        start_date=start_date,
        end_date=end_date,
        tune_hyperparameters=False,
        plot_results=False,
        save_assets=True,
        use_tuned_strategy_params=True,
        use_tuned_hyper_params=True
    )
    print(f"total duration: {datetime.now() - start}")
    print(f"all_quality_scores={all_quality_scores}")
    print_full(result.metrics_df)
    return result

def run_backtest_portfolio(portfolio_name, tickers, strategy_type, start_date, end_date):
    result = run_pybroker_portfolio_backtest(
        portfolio_name=portfolio_name,
        tickers=tickers,
        strategy_type = strategy_type,
        start_date=start_date,
        end_date=end_date,
        plot_results= False,
        use_tuned_strategy_params= True,    #  ./WORK/strategy_configs/{strategy_type}.json
        model_mode = ModelMode.TRAINED,     #  ./WORK/strategy_configs/{strategy_type}_model.pkl
        max_open_positions= 10,
        commission_cost = 0.0)
    print_full(result.metrics_df)
    return result


def run_pipeline_tune_train_backtest(
        portfolio_name,
        tickers, strategy_type, start_date, end_date,
        backtest_portfolio_name,
        backtest_tickers, backtest_start_date, backtest_end_date):
    start = datetime.now()

    # 1. tune strategy parameters and copy  _best_strategy_params.json to strategy_configs directory
    run_tune_strategy(portfolio_name, tickers, strategy_type, start_date, end_date)
    copy_to_strategy_configs(f"{portfolio_name}_{strategy_type}_best_strategy_params.json", f"{strategy_type}.json")
    is_ml = is_ml_strategy(strategy_type)
    if is_ml :
        # 2. tune hyperparameters and copy _best_params.json to strategy_configs directory
        run_tune_hyper_params(portfolio_name, tickers, strategy_type, start_date, end_date)
        copy_to_strategy_configs(f"{portfolio_name}_{strategy_type}_best_params.json", f"{strategy_type}_hyper_params.json")

        # 3. train model and copy .pkl file to strategy_configs directory
        run_train_model(portfolio_name, tickers, strategy_type, start_date, end_date)
        copy_to_strategy_configs(f"{portfolio_name}_{strategy_type}.pkl", f"{strategy_type}_model.pkl")
    else:
        print(f"Not ML strategy for {strategy_type}, so skip Tune Hyperparams and Train Model")

    # 4. use the tuned strategy_parameter, trained model to run backtest out of train window to see result
    run_backtest_portfolio(backtest_portfolio_name, backtest_tickers, strategy_type, backtest_start_date, backtest_end_date)

    print(f"start = {start}, total duration: {datetime.now() - start}")

