import bt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class SelectWhere(bt.Algo):
    def __init__(self, signal):
        self.signal = signal

    def __call__(self, target):
        if target.now in self.signal.index:
            sig = self.signal.loc[target.now]

            selected = list(sig.index[sig])

            target.temp['selected'] = selected
        return True


class WeighTarget(bt.Algo):

    def __init__(self, target_weights):
        self.tw = target_weights

    def __call__(self, target):
        # get target weights on date target.now
        if target.now in self.tw.index:
            w = self.tw.loc[target.now]

            # save in temp - this will be used by the weighing algo
            # also dropping any na's just in case they pop up
            target.temp['weights'] = w.dropna()

        # return True because we want to keep on moving down the stack
        return True


def above_sma(tickers, sma_per=50, start='2010-01-01', name='above_sma'):
    """
    Long securities that are above their n period
    Simple Moving Averages with equal weights.
    """
    # download data
    data = bt.get(tickers, start=start)
    # calc sma
    sma = data.rolling(sma_per).mean()

    # create strategy
    s = bt.Strategy(name, [SelectWhere(data > sma),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])

    # now we create the backtest
    return bt.Backtest(s, data)


## download data
data = bt.get('spy:Open, spy:Close', start='2010-01-01')
data2 = bt.get('spy', start='2010-01-01')

## manipulate data
w = pd.DataFrame(data['spyclose'] / data['spyopen'], columns=['spy']).shift(periods=1)
w2 = -np.log(w) * 10

print(w2)
# ax = w2.plot()

## create strategy

s = bt.Strategy('Mean Reversion', [WeighTarget(w2),
                                   bt.algos.RebalanceOverTime()])
t = bt.Backtest(s, data2)
res = bt.run(t)
res.plot()
res.display()

plt.show()
