
All of those py files are extracted from quant_engine.py, and first refactored to Walkforward class.
The new Workforward class are able to accept a portfolio (multiple tickers) to tune and train,
the class structure is easier/more flexible to extend.
In theory, training a single model on multiple tickers (often called Global Modeling or Cross-Sectional Training)
is a common practice to give the model more data points and help it learn general market patterns.

In the pipeline,
def run_pipeline_tune_train_backtest(
        portfolio_name,
        tickers, strategy_type, start_date, end_date,
        backtest_portfolio_name,
        backtest_tickers, backtest_start_date, backtest_end_date):
    start = datetime.now()
    # 1. tune strategy parameters and copy  _best_strategy_params.json to strategy_configs directory
    run_tune_strategy(portfolio_name, tickers, strategy_type, start_date, end_date)
    copy_to_strategy_configs(f"{portfolio_name}_{strategy_type}_best_strategy_params.json", f"{strategy_type}.json")

    # 2. tune hyperparameters and copy _best_params.json to strategy_configs directory
    run_tune_hyper_params(portfolio_name, tickers, strategy_type, start_date, end_date)
    copy_to_strategy_configs(f"{portfolio_name}_{strategy_type}_best_params.json", f"{strategy_type}_hyper_params.json")

    # 3. train model and copy .pkl file to strategy_configs directory
    run_train_model(portfolio_name, tickers, strategy_type, start_date, end_date)
    copy_to_strategy_configs(f"{portfolio_name}_{strategy_type}.pkl", f"{strategy_type}_model.pkl")

    # 4. use the tuned strategy_parameter, trained model to run backtest out of train window to see result
    run_backtest_portfolio(backtest_portfolio_name, backtest_tickers, strategy_type, backtest_start_date, backtest_end_date)


The above provide a structure/framework/pipeline as a starting point, we can try difference portfolios, model configurations(scaler),
strategy/features parameters and ... more. and use the results/metrix to evaluate and improve overall ML performance.

You can run run_backtest(use TRAINED model), and run_backtest_passthrough(use PASSTHROUGH model) to compare results.

or you can run run_backtest_compare.py, if you have already trained 'structure_liquidity' and 'donchian_breakout',
it will compare [TRAINED, PASSTHROUGH(No Train), No Tune] with combination of two strategies, three tickers set,
print out concat metrix_df result.

    tickers1 = ['SPY', 'WMT', 'T', 'JPM', 'BAC', 'C', 'CAT', 'FDX',
               'PFE', 'COST', 'AMZN', 'AAPL', 'INTC', 'DIS', 'HD',
               'NFLX', 'UNH', 'PG', 'KO', 'CSCO', 'BA']

    tickers2 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'NFLX', 'ADBE',
     'CRM', 'NOW', 'ASML', 'TSM', 'CAT', 'DE', 'UNP', 'JPM', 'GS', 'AAL', 'JNJ',
     'UNH', 'PG', 'KO', 'NEE', 'DUK', 'PLUG', 'RIVN', 'HOOD', 'DNN']

    tickers3 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'NOW', 'ASML', 'TSM',
     'CAT', 'DE', 'UNP', 'JPM', 'GS', 'AAL', 'JNJ', 'UNH', 'PG', 'KO', 'NEE', 'DUK', 'PLUG', 'RIVN', 'HOOD', 'DNN',
     'AMD', 'INTC', 'CSCO', 'ORCL', 'PFE', 'MRK', 'DIS', 'BAC', 'WMT', 'HD', 'XOM', 'CVX', 'NKE', 'MCD', 'T', 'SOFI',
     'CCL', 'GM', 'PYPL']

==========================================================================

====== structure_liquidity-ticker1-2021-01-01-2025-12-31 ========
                      name        trained value       no train value  no tune/train value
0              trade_count                   71                   91                  178
1     initial_market_value             100000.0             100000.0             100000.0
2         end_market_value            157452.02            135229.64            122257.89
3                total_pnl             57915.82             35681.12             22596.09
4           unrealized_pnl               -463.8              -451.48               -338.2
5         total_return_pct             57.91582             35.68112             22.59609
6             total_profit             90048.51             71863.99            117920.29
7               total_loss            -32132.69            -36182.87             -95324.2
8               total_fees                  0.0                  0.0                  0.0
9             max_drawdown            -17144.94            -16925.25             -25582.3
10        max_drawdown_pct           -11.538373           -13.481426             -17.8853
11       max_drawdown_date  2025-04-08 00:00:00  2023-10-27 00:00:00  2025-11-20 00:00:00
12                win_rate            43.661972            42.857143            44.382022
13               loss_rate            56.338028            57.142857            55.617978
14          winning_trades                   31                   39                   79
15           losing_trades                   40                   52                   99
16                 avg_pnl           815.715775            392.10022           126.944326
17          avg_return_pct             7.047183             5.550659             0.609831
18          avg_trade_bars            66.957746            59.901099            16.988764
19              avg_profit          2904.790645           1842.66641          1492.661899
20          avg_profit_pct            24.115484            20.984103             6.780886
21  avg_winning_trade_bars           119.677419           111.615385            26.974684
22                avg_loss           -803.31725          -695.824423          -962.870707
23            avg_loss_pct             -6.18075            -6.024423            -4.314545
24   avg_losing_trade_bars                 26.1            21.115385             9.020202
25             largest_win             16125.12              6546.77             14590.49
26         largest_win_pct                76.97                51.66                35.97
27        largest_win_bars                  268                  112                   51
28            largest_loss             -1667.29             -1320.73             -4870.45
29        largest_loss_pct                -5.25                -7.98                -13.5
30       largest_loss_bars                   17                    6                    7
31                max_wins                    3                    5                   11
32              max_losses                    7                   11                    8
33                  sharpe             0.109073             0.066869             0.035995
34                 sortino              0.15796             0.091273             0.054005
35           profit_factor             1.317489             1.188758             1.081349
36             ulcer_index             1.714928             2.118638             3.074324
37                     upi             0.043676             0.024145             0.012209
38               equity_r2             0.954423             0.913395             0.526871
39               std_error           21605.8065         12284.827762          11459.95253

====== structure_liquidity-ticker2-2021-01-01-2025-12-31 ========
                      name        trained value       no train value  no tune/train value
0              trade_count                   95                   90                  224
1     initial_market_value             100000.0             100000.0             100000.0
2         end_market_value            180963.99            198203.93            189290.31
3                total_pnl             81501.95             98853.58             90497.01
4           unrealized_pnl              -537.96              -649.65              -1206.7
5         total_return_pct             81.50195             98.85358             90.49701
6             total_profit            134072.07             138398.2            193724.88
7               total_loss            -52570.12            -39544.62           -103227.87
8               total_fees                  0.0                  0.0                  0.0
9             max_drawdown            -27177.97            -21255.35            -33142.28
10        max_drawdown_pct           -16.942514           -11.926736           -17.982818
11       max_drawdown_date  2025-04-08 00:00:00  2025-04-08 00:00:00  2025-04-04 00:00:00
12                win_rate            48.421053            54.444444            40.178571
13               loss_rate            51.578947            45.555556            59.821429
14          winning_trades                   46                   49                   90
15           losing_trades                   49                   41                  134
16                 avg_pnl           857.915263          1098.373111           404.004509
17          avg_return_pct            11.556316            12.139556             2.709464
18          avg_trade_bars            61.094737            66.655556            18.674107
19              avg_profit          2914.610217          2824.453061          2152.498667
20          avg_profit_pct            34.664783            29.854286            15.363111
21  avg_winning_trade_bars           105.543478           105.795918            31.066667
22                avg_loss         -1072.859592          -964.502927          -770.357239
23            avg_loss_pct           -10.137347            -9.031707            -5.789254
24   avg_losing_trade_bars            19.367347            19.878049            10.350746
25             largest_win             12449.58             12448.61             16979.39
26         largest_win_pct               104.05               267.91                40.95
27        largest_win_bars                  182                  133                   66
28            largest_loss             -2246.92              -2083.7              -4275.0
29        largest_loss_pct               -13.59                -9.68                -8.16
30       largest_loss_bars                   17                   23                    9
31                max_wins                   10                    8                   10
32              max_losses                    8                   10                   10
33                  sharpe             0.103549             0.122834             0.086376
34                 sortino             0.148648             0.190004             0.126607
35           profit_factor             1.305814             1.369778             1.237357
36             ulcer_index             2.411305              2.16297             3.067167
37                     upi             0.041223             0.052526             0.035894
38               equity_r2             0.922574             0.974335              0.92444
39               std_error         25485.167832         33370.935752         36032.205804

====== structure_liquidity-ticker3-2021-01-01-2025-12-31 ========
                      name        trained value       no train value  no tune/train value
0              trade_count                   92                  106                  271
1     initial_market_value             100000.0             100000.0             100000.0
2         end_market_value            204400.18            205885.99            130013.58
3                total_pnl            104896.48            106351.22             30640.28
4           unrealized_pnl               -496.3              -465.23               -626.7
5         total_return_pct            104.89648            106.35122             30.64028
6             total_profit            150919.76            155813.09            145167.86
7               total_loss            -46023.28            -49461.87           -114527.58
8               total_fees                  0.0                  0.0                  0.0
9             max_drawdown            -26417.55            -25475.89            -38596.83
10        max_drawdown_pct           -14.805578           -13.919656           -26.765169
11       max_drawdown_date  2025-04-08 00:00:00  2025-04-08 00:00:00  2025-04-08 00:00:00
12                win_rate                 50.0            51.886792            40.740741
13               loss_rate                 50.0            48.113208            59.259259
14          winning_trades                   46                   55                  110
15           losing_trades                   46                   51                  160
16                 avg_pnl           1140.17913          1003.313396           113.063764
17          avg_return_pct            12.535761             10.94066             1.005055
18          avg_trade_bars            65.032609            57.367925            18.332103
19              avg_profit          3280.864348          2832.965273          1319.707818
20          avg_profit_pct            34.041957            28.564727            11.058091
21  avg_winning_trade_bars                110.0                 92.0            31.254545
22                avg_loss         -1000.506087          -969.840588          -715.797375
23            avg_loss_pct            -8.970435            -8.065686            -5.900125
24   avg_losing_trade_bars            20.065217            20.019608               9.5625
25             largest_win              12068.5             12543.64             12112.69
26         largest_win_pct               267.91               267.91                40.95
27        largest_win_bars                  133                  133                   66
28            largest_loss             -1869.07             -1996.96             -2841.98
29        largest_loss_pct               -10.05                -8.67               -13.22
30       largest_loss_bars                    9                   18                    5
31                max_wins                    7                    9                    5
32              max_losses                    7                    7                   15
33                  sharpe             0.122914             0.124088             0.040851
34                 sortino             0.179124             0.181387             0.057927
35           profit_factor             1.373397              1.36453             1.091974
36             ulcer_index             2.352581              2.26751             3.648079
37                     upi             0.050557             0.052969             0.013488
38               equity_r2             0.906434             0.968688             0.637815
39               std_error         30141.107589         34222.224954         13255.665696

====== donchian_breakout-ticker1-2021-01-01-2025-12-31 ========
                      name        trained value       no train value  no tune/train value
0              trade_count                   80                  157                  116
1     initial_market_value             100000.0             100000.0             100000.0
2         end_market_value            176906.58            154420.28            130919.49
3                total_pnl             77072.31             54959.84             31407.02
4           unrealized_pnl              -165.73              -539.56              -487.53
5         total_return_pct             77.07231             54.95984             31.40702
6             total_profit            106223.94            117263.61             81139.03
7               total_loss            -29151.63            -62303.77            -49732.01
8               total_fees                  0.0                  0.0                  0.0
9             max_drawdown             -8953.92            -16728.51            -18797.34
10        max_drawdown_pct            -6.312824           -11.181032           -15.562978
11       max_drawdown_date  2024-04-03 00:00:00  2023-10-27 00:00:00  2023-10-27 00:00:00
12                win_rate                 55.0            46.496815            42.241379
13               loss_rate                 45.0            53.503185            57.758621
14          winning_trades                   44                   73                   49
15           losing_trades                   36                   84                   67
16                 avg_pnl           963.403875           350.062675           270.750172
17          avg_return_pct                2.375             0.988854             1.463534
18          avg_trade_bars                16.15            14.757962            21.948276
19              avg_profit          2414.180455          1606.350822          1655.898571
20          avg_profit_pct             6.389318             5.501096             7.604898
21  avg_winning_trade_bars            21.863636            21.890411            38.673469
22                avg_loss            -809.7675          -741.711548          -742.268806
23            avg_loss_pct            -2.531389              -2.9325             -3.02791
24   avg_losing_trade_bars             9.166667             8.559524             9.716418
25             largest_win              8380.19              7767.21             11268.15
26         largest_win_pct                14.02                 16.3                43.21
27        largest_win_bars                   48                   55                  124
28            largest_loss             -2852.53             -2834.19             -3726.06
29        largest_loss_pct                -5.52                 -4.3                -5.52
30       largest_loss_bars                    4                    1                    1
31                max_wins                    5                    9                    8
32              max_losses                    5                   10                   10
33                  sharpe             0.124648             0.084519             0.050515
34                 sortino             0.182077             0.130107             0.074505
35           profit_factor             1.456231             1.240725             1.136629
36             ulcer_index             1.665093             2.055006             2.470505
37                     upi             0.056449             0.035607             0.019204
38               equity_r2             0.956652             0.917059             0.781121
39               std_error         22028.092737         20845.973425         12651.303395

====== donchian_breakout-ticker2-2021-01-01-2025-12-31 ========
                      name        trained value       no train value  no tune/train value
0              trade_count                  118                  207                  168
1     initial_market_value             100000.0             100000.0             100000.0
2         end_market_value            177163.94            157083.77            111303.97
3                total_pnl              77784.2             57500.96             11899.21
4           unrealized_pnl              -620.26              -417.19              -595.24
5         total_return_pct              77.7842             57.50096             11.89921
6             total_profit            140701.85            143403.07             87555.57
7               total_loss            -62917.65            -85902.11            -75656.36
8               total_fees                  0.0                  0.0                  0.0
9             max_drawdown            -14843.97            -17516.06             -28400.1
10        max_drawdown_pct           -11.439799           -16.610463           -25.405432
11       max_drawdown_date  2024-11-01 00:00:00  2023-10-27 00:00:00  2025-04-07 00:00:00
12                win_rate            48.305085            45.410628            41.071429
13               loss_rate            51.694915            54.589372            58.928571
14          winning_trades                   57                   94                   69
15           losing_trades                   61                  113                   99
16                 avg_pnl           659.188136           277.782415            70.828631
17          avg_return_pct             2.080847             2.152657             2.546786
18          avg_trade_bars            14.864407             13.84058            18.886905
19              avg_profit          2468.453509          1525.564574          1268.921304
20          avg_profit_pct              8.67193             9.874362            12.944348
21  avg_winning_trade_bars            22.192982            21.234043             32.26087
22                avg_loss         -1031.436885          -760.195664          -764.205657
23            avg_loss_pct            -4.078033            -4.270708                 -4.7
24   avg_losing_trade_bars             8.016393             7.690265             9.565657
25             largest_win             13913.51              9589.92             11212.91
26         largest_win_pct                27.91                85.42               106.41
27        largest_win_bars                   21                   48                   59
28            largest_loss             -3758.04              -2659.0             -2858.54
29        largest_loss_pct                -9.75                -4.71               -11.26
30       largest_loss_bars                    4                   11                    4
31                max_wins                    6                    8                    5
32              max_losses                    6                    9                   11
33                  sharpe             0.100277             0.072744             0.021196
34                 sortino             0.151004             0.105424             0.028117
35           profit_factor             1.338742             1.222897              1.04721
36             ulcer_index             2.331691             2.725833             3.309164
37                     upi             0.041158             0.028576              0.00696
38               equity_r2             0.830264             0.764557             0.110435
39               std_error         20764.966244         18829.283333          9071.340731

====== donchian_breakout-ticker3-2021-01-01-2025-12-31 ========
                      name        trained value       no train value  no tune/train value
0              trade_count                  155                  267                  217
1     initial_market_value             100000.0             100000.0             100000.0
2         end_market_value             166098.7            150686.13             86509.66
3                total_pnl             66721.48             51522.55            -13105.59
4           unrealized_pnl              -622.78              -836.42              -384.75
5         total_return_pct             66.72148             51.52255            -13.10559
6             total_profit            146099.49            130016.21              71495.3
7               total_loss            -79378.01            -78493.66            -84600.89
8               total_fees                  0.0                  0.0                  0.0
9             max_drawdown             -16830.2             -23258.5            -28064.81
10        max_drawdown_pct           -12.436739           -18.492323           -26.448283
11       max_drawdown_date  2025-05-06 00:00:00  2023-10-27 00:00:00  2024-07-25 00:00:00
12                win_rate            43.225806            43.071161             36.40553
13               loss_rate            56.774194            56.928839             63.59447
14          winning_trades                   67                  115                   79
15           losing_trades                   88                  152                  138
16                 avg_pnl           430.461161           192.968352           -60.394424
17          avg_return_pct             1.503161             1.295019             1.019078
18          avg_trade_bars            13.116129            12.891386            16.576037
19              avg_profit          2180.589403          1130.575739           905.003797
20          avg_profit_pct             8.689851             7.977217            10.551282
21  avg_winning_trade_bars             20.41791            19.582609            29.240506
22                avg_loss          -902.022841          -516.405658          -613.049928
23            avg_loss_pct            -3.968523            -3.760592            -4.361304
24   avg_losing_trade_bars             7.556818             7.828947             9.326087
25             largest_win             12588.42              7670.49              7447.88
26         largest_win_pct                27.91                51.78                 19.7
27        largest_win_bars                   21                   45                   78
28            largest_loss             -3758.04             -2975.59             -2260.44
29        largest_loss_pct                -9.75               -10.96                -3.65
30       largest_loss_bars                    4                    2                   11
31                max_wins                    6                    6                    7
32              max_losses                    7                    8                    9
33                  sharpe             0.087338             0.065955            -0.015944
34                 sortino             0.144153              0.10225            -0.022569
35           profit_factor             1.274247             1.204859             0.940464
36             ulcer_index             2.528648              2.73869             3.645123
37                     upi             0.033971             0.026053            -0.004738
38               equity_r2             0.723935             0.761679             0.028091
39               std_error          16901.82024         18958.356939          5250.472923

==========================================================================
