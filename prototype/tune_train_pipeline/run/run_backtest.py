import pandas as pd

from prototype.tune_train_pipeline.pipeline_tune_train_test import run_backtest_portfolio


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


    result = run_backtest_portfolio(
        portfolio_name='backtest',
        tickers=tickers2,
        strategy_type =  'structure_liquidity',   #'structure_liquidity',  #'donchian_breakout',
        start_date='2021-01-01',
        end_date='2025-12-31')

