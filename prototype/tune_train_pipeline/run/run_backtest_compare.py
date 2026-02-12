import pandas as pd

from prototype.tune_train_pipeline.backtest_portfolio import run_pybroker_portfolio_backtest, ModelMode
from prototype.tune_train_pipeline.pipeline_tune_train_test import run_backtest_portfolio
from prototype.tune_train_pipeline.tune_train_base import print_full, get_all_strategy_types, is_ml_strategy


def merge_df(trained_df, df, no_tune_df):
    if trained_df is not None:
        trained_df = trained_df.rename(columns={'value': 'trained value'})
        df = df.rename(columns={'value': 'no train value'})
        no_tune_df = no_tune_df.rename(columns={'value': 'no tune/train value'})
    else:
        df = df.rename(columns={'value': 'tuned value'})
        no_tune_df = no_tune_df.rename(columns={'value': 'no tune value'})

    if trained_df is None:
        return pd.merge(df, no_tune_df, on='name')
    else:
        return pd.merge(pd.merge(trained_df, df, on='name'), no_tune_df, on='name')

def run_backtest_compare(portfolio_name, tickers, strategy_type, start_date, end_date):
    is_ml = is_ml_strategy(strategy_type)
    print(f"====== TRAINED Model BACKTEST {portfolio_name} ========")
    trained_result = None
    if is_ml:
        trained_result = run_backtest_portfolio(
            portfolio_name=portfolio_name,
            tickers=tickers,
            strategy_type=strategy_type,
            start_date=start_date,
            end_date=end_date)

    print(f"====== No Train Model BACKTEST {portfolio_name} ========")
    result = run_pybroker_portfolio_backtest(
        portfolio_name=portfolio_name,
        tickers=tickers,
        strategy_type=strategy_type,
        start_date=start_date,
        end_date=end_date,
        plot_results=False,
        use_tuned_strategy_params=True,  # ./WORK/strategy_configs/{strategy_type}.json
        model_mode=ModelMode.PASSTHROUGH,  # ./WORK/strategy_configs/{strategy_type}_model.pkl
        max_open_positions=10,
        commission_cost=0.0)
    print_full(result.metrics_df)

    print(f"====== No Tune No Train BACKTEST {portfolio_name} ========")
    no_tune_result = run_pybroker_portfolio_backtest(
        portfolio_name=portfolio_name,
        tickers=tickers,
        strategy_type=strategy_type,
        start_date=start_date,
        end_date=end_date,
        plot_results=False,
        use_tuned_strategy_params=False,
        model_mode=ModelMode.PASSTHROUGH,  # ./WORK/strategy_configs/{strategy_type}_model.pkl
        max_open_positions=10,
        commission_cost=0.0)
    print_full(no_tune_result.metrics_df)

    compare_df = merge_df(None if trained_result is None else trained_result.metrics_df, result.metrics_df, no_tune_result.metrics_df)


    print("\n")
    print_full(compare_df)
    return compare_df


if __name__ == "__main__":
    tickers1 = ['SPY', 'WMT', 'T', 'JPM', 'BAC', 'C', 'CAT', 'FDX',
                'PFE', 'COST', 'AMZN', 'AAPL', 'INTC', 'DIS', 'HD',
                'NFLX', 'UNH', 'PG', 'KO', 'CSCO', 'BA']

    tickers2 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'NFLX', 'ADBE',
                'CRM', 'NOW', 'ASML', 'TSM', 'CAT', 'DE', 'UNP', 'JPM', 'GS', 'AAL', 'JNJ',
                'UNH', 'PG', 'KO', 'NEE', 'DUK', 'PLUG', 'RIVN', 'HOOD', 'DNN']

    tickers3 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'NOW', 'ASML',
                'TSM',
                'CAT', 'DE', 'UNP', 'JPM', 'GS', 'AAL', 'JNJ', 'UNH', 'PG', 'KO', 'NEE', 'DUK', 'PLUG', 'RIVN', 'HOOD',
                'DNN',
                'AMD', 'INTC', 'CSCO', 'ORCL', 'PFE', 'MRK', 'DIS', 'BAC', 'WMT', 'HD', 'XOM', 'CVX', 'NKE', 'MCD', 'T',
                'SOFI',
                'CCL', 'GM', 'PYPL']
    tickers_list = [tickers1, tickers2, tickers3]

    # strategy_types = ['donchian_breakout', 'structure_liquidity', 'ma_crossover', 'structure_liquidity0']
    strategy_types = get_all_strategy_types()
    start_date = '2021-01-01'
    end_date = '2025-12-31'

    compare_dfs = dict()
    for strategy_type in strategy_types:
        for i, tickers in enumerate(tickers_list):
            portfolio_name = f"{strategy_type}-ticker{i + 1}-{start_date}-{end_date}"
            df = run_backtest_compare(
                portfolio_name=portfolio_name,
                tickers=tickers,
                strategy_type=strategy_type,
                start_date=start_date,
                end_date=end_date
            )
            compare_dfs[portfolio_name] = df

    print(f"\n\n==========================================================================")
    for portfolio_name, df in compare_dfs.items():
        print(f"\n====== {portfolio_name} ========")
        print_full(df)
    print(f"\n==========================================================================")
