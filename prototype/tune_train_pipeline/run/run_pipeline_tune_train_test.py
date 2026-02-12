from prototype.tune_train_pipeline.pipeline_tune_train_test import run_pipeline_tune_train_backtest
from prototype.tune_train_pipeline.tune_train_base import train_tickers, backtest_tickers

if __name__ == '__main__':
    # strategy_types = ['donchian_breakout', 'structure_liquidity', 'ma_crossover', 'structure_liquidity0']
    run_pipeline_tune_train_backtest(
        portfolio_name='25_stocks',
        tickers=train_tickers,
        strategy_type=   'donchian_breakout',
        start_date='2000-01-01',
        end_date='2020-01-01',
        backtest_portfolio_name='backtest',
        backtest_tickers=backtest_tickers,
        backtest_start_date= '2021-01-01',
        backtest_end_date='2025-12-30'
    )
