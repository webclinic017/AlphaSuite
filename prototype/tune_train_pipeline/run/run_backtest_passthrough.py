import pandas as pd

from prototype.tune_train_pipeline.backtest_portfolio import run_pybroker_portfolio_backtest, ModelMode
from prototype.tune_train_pipeline.pipeline_tune_train_test import run_backtest_portfolio
from prototype.tune_train_pipeline.tune_train_base import print_full

if __name__ == "__main__":
    tickers = ['SPY', 'WMT', 'T', 'JPM', 'BAC', 'C', 'CAT', 'FDX',
               'PFE', 'COST', 'AMZN', 'AAPL', 'INTC', 'DIS', 'HD',
               'NFLX', 'UNH', 'PG', 'KO', 'CSCO', 'BA']

    tickers1 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'NFLX', 'ADBE',
     'CRM', 'NOW', 'ASML', 'TSM', 'CAT', 'DE', 'UNP', 'JPM', 'GS', 'AAL', 'JNJ',
     'UNH', 'PG', 'KO', 'NEE', 'DUK', 'PLUG', 'RIVN', 'HOOD', 'DNN']

    tickers2 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'NOW', 'ASML', 'TSM',
     'CAT', 'DE', 'UNP', 'JPM', 'GS', 'AAL', 'JNJ', 'UNH', 'PG', 'KO', 'NEE', 'DUK', 'PLUG', 'RIVN', 'HOOD', 'DNN',
     'AMD', 'INTC', 'CSCO', 'ORCL', 'PFE', 'MRK', 'DIS', 'BAC', 'WMT', 'HD', 'XOM', 'CVX', 'NKE', 'MCD', 'T', 'SOFI',
     'CCL', 'GM', 'PYPL']


    result = run_pybroker_portfolio_backtest(
        portfolio_name='backtest',
        tickers=tickers2,
        strategy_type =  'structure_liquidity', #'structure_liquidity',  #'donchian_breakout',
        start_date='2021-01-01',
        end_date='2025-12-31',
        plot_results=False,
        use_tuned_strategy_params=True,  # ./WORK/strategy_configs/{strategy_type}.json
        model_mode=ModelMode.PASSTHROUGH,  # ./WORK/strategy_configs/{strategy_type}_model.pkl
        max_open_positions=10,
        commission_cost=0.0)

    print_full(result.metrics_df)
