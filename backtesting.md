# Backtesting

- The backtesting module implements the following.
- We keep track of the 'testing date', which moves forward in time from the past and applies a selected
strategy at that time. First, we select the Universe using the universe condition.
Next, we buy and sell using the buy condition and the sell condition. However, for the sake of simplicity, we 
sell everything the very next day.
- To buy or sell, for each 'testing date', we keep track of the order, deposit, and balance.
- At the end of the test, we produce a performance summary of the strategy. The summary includes:
  - Sharpe (= avg annual Return / avg annual stddev of Return)
  - Turnover (= Avg value traded / value held)
  - Fitness (= Sharpe * sqrt(abs(Returns)) / max(Turnover, 0.125))
  - Returns (= Avg profit or loss / invested amount * 2)
  - Drawdown (= Largest reduction of profit and loss)
  - Margin (= Avg profit or loss / amount traded)
  - Cumulative Profit and Loss graph
- The summary can be calculated by saving the data of daily deposit, daily sold value, and daily bought value.

Although this backtesting module implements a very simple sell condition, it tests the validity of the buy condition
assuming that the average performance converges after testing over long period of time.