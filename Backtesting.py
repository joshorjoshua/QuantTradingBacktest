import collections

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from util.db_helper import *
from util.const import *
from util.time_helper import *
from util.np_helper import *
from datetime import *
import talib
import math
import cmath


class Backtesting():

    def __init__(self):
        self.universe = {}
        self.universe_close = collections.defaultdict(dict)

        self.start_deposit = 10000000
        self.deposit = self.start_deposit
        self.balance = {}

        # Adjust based on market sentiment.
        self.investment_deposit_ratio = 0.9  # [0, 1]
        self.investment_ratio = 0.4  # [0, 1]
        self.stop_loss = 0.06

        self.start_date = date(1985, 1, 1)  # .strftime('%Y%m%d')
        self.testing_date = self.start_date

        self.weight = {}

        self.summary = {}

        self.init()

    def init(self):
        for code in codes:
            self.universe[code] = {'code_name': code}

        self.get_data()

        for code in codes:
            for dt in self.universe[code]['price_df'].index:
                self.universe_close[dt][code] = self.universe[code]['price_df'].loc[dt, 'close']

            self.balance[code] = {
                'invested': 0,
                'quantity': 0
            }

    def alpha(self, code):
        dt = self.testing_date.strftime("%Y%m%d")
        universe_item = self.universe[code]

        high = self.universe[code]['price_df']['high']
        low = self.universe[code]['price_df']['low']
        open = self.universe[code]['price_df']['open']
        close = self.universe[code]['price_df']['close']
        volume = self.universe[code]['price_df']['volume']

        # _____________COMMON INDICATORS_________________

        # simple moving average
        sma = talib.SMA(close, 200)

        # Bollinger Bands
        upper, mid, lower = talib.BBANDS(close,
                                         nbdevup=2,
                                         nbdevdn=2,
                                         timeperiod=20)

        # MACD
        # MACD_hist > 0 indicates bullish market
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        # Stochastic (d is 3-day moving average of k)
        # k < 20 indicates buy signal
        # k peaks just below 100 indicates sell signal
        # k rises above d and k < 80 indicates buy signal
        k, d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0,
                           slowd_period=3, slowd_matype=0)
        stoch_hist = k - d

        # Parabolic SAR
        psar = talib.SAR(high, low)

        # ADX
        adx = talib.ADX(high, low, close)

        # _______________STRATEGY_______________________
        """
        # MACD + Stochastic double-cross strategy
        # 1. (0.5) Positive macd_hist and stoch_hist
        # 2. (0.4) Bullish crossover within 2 days (Stochastic must cross before MACD)
        # 3. (0.05) k < 50
        # 4. (0.05) Above 200 day m.a.

        # 1.
        c = macd_hist[dt] > 0 and stoch_hist[dt] > 0

        # 2.
        c1 = False  # MACD cross-over
        if str_timedelta(dt, -1) in macd_hist:
            c1 = macd_hist[dt] > 0 and macd_hist[str_timedelta(dt, -1)] > 0

        c2 = False  # Stochastic cross-over
        for i in range(2):
            if c2:
                break
            if str_timedelta(dt, -i) not in stoch_hist or str_timedelta(dt, -(i + 1)) not in stoch_hist:
                continue
            c2 = c2 or (stoch_hist[str_timedelta(dt, -i)] > 0 > stoch_hist[str_timedelta(dt, -(i + 1))])

        # 3.
        c3 = k[dt] < 50

        # 4.
        c4 = close[dt] > sma[dt]
        """

        # PSAR strategy
        # 1. (0.4) PSAR < close
        # 2. (0.3) PSAR buy signal
        # 3. (0.3 if 2) adx > 30

        # 1.
        c1 = psar[dt] < close[dt]

        # 2.
        c2 = False  # MACD cross-over
        if str_timedelta(dt, -1) in psar:
            c2 = psar[dt] < close[dt] and psar[str_timedelta(dt, -1)] < close[dt]

        # 3.
        c3 = adx[dt] > 30

        # Update weight
        self.weight[code] = c1 * 0.4 + c2 * 0.3 + (c2 and c3) * 0.3


    def get_data(self):
        for idx, code in enumerate(self.universe.keys()):
            sql = "select * from `{}`".format(code)
            cur = execute_sql("BollingerStrategy", sql)
            cols = [column[0] for column in cur.description]

            price_df = pd.DataFrame.from_records(data=cur.fetchall(), columns=cols)
            price_df = price_df.set_index('index')

            self.universe[code]['price_df'] = price_df

    def print_data(self):
        for idx, code in enumerate(self.universe.keys()):
            print(self.universe[code]['price_df'].head())

    def run(self):
        while self.testing_date < date.today():
            dt = self.testing_date.strftime("%Y%m%d")
            sample_code = '000270'
            # skip weekends when market is closed
            if dt not in self.universe[sample_code]['price_df'].index:
                self.testing_date += timedelta(days=1)
                continue
            if dt not in self.universe_close:
                self.testing_date += timedelta(days=1)
                continue

            bought = 0
            sold = 0
            in_stock = 0
            daily_deposit = self.deposit
            print("Testing at: " + dt)

            # update weight for this date
            for code in codes:
                if dt not in self.universe[code]['price_df'].index:
                    continue
                self.alpha(code)

            # Buy / Sell
            for code in codes:
                if dt not in self.universe[code]['price_df'].index:
                    continue

                current_quantity = self.balance[code]['quantity']
                optimal_quantity = self.get_quantity(code, daily_deposit)

                if self.buy_condition(code):
                    # buy
                    if current_quantity < optimal_quantity:
                        bought += self.buy(code, optimal_quantity - current_quantity)
                if self.sell_condition(code):
                    # sell
                    if current_quantity > optimal_quantity:
                        sold += self.sell(code, current_quantity - optimal_quantity)

            # Update Summary
            for code in codes:
                if code not in self.balance:
                    continue
                if code not in self.universe_close[dt]:
                    continue
                # Calculate value in stock if it were to be sold today.
                in_stock += self.balance[code]['quantity'] * self.universe_close[dt][code]

            print("Deposit: " + str(self.deposit))
            print("In stock: " + str(in_stock))
            print("Total: " + str(self.deposit + in_stock))

            self.summary[self.testing_date] = {
                'deposit': self.deposit,
                'in stock': in_stock,
                'bought': bought,
                'sold': sold
            }

            self.testing_date += timedelta(days=1)

        self.print_summary()

    def get_quantity(self, code, daily_deposit):
        dt = self.testing_date.strftime("%Y%m%d")
        close = self.universe[code]['price_df'].loc[dt, 'close']

        # stop loss
        invested = self.balance[code]['invested']
        stocks = self.balance[code]['quantity']
        avg_price = 0
        if stocks > 0:
            avg_price = invested / stocks
        if close < avg_price * (1 - self.stop_loss):
            return 0

        # Calculate optimal quantity from weights
        nw = self.normalized_weight(code)

        quantity = int(round(np.nan_to_num(daily_deposit / (200 * self.investment_ratio) / close * nw)))

        # cannot short a stock, so minimum quantity is 0
        if quantity < 0:
            quantity = 0

        return quantity

    def normalized_weight(self, code):
        w = np.array(list(self.weight.values()))

        min = w.min()
        max = w.max()

        # scale every value into [0, 1]
        ans = (self.weight[code] - min) / (max - min)

        # normalize
        ans = (ans - (1 - self.investment_ratio)) / self.investment_ratio

        return ans

    def buy_condition(self, code):
        dt = self.testing_date.strftime("%Y%m%d")
        universe_item = self.universe[code]

        return True

    def sell_condition(self, code):
        if code not in self.balance:
            return False

        return True

    def buy(self, code, quantity):
        if quantity < 1:
            return 0

        dt = self.testing_date.strftime("%Y%m%d")
        price = self.universe[code]['price_df'].loc[dt, 'close']

        if quantity * price > self.deposit:
            return 0

        # subtract from deposit and update balance
        self.deposit -= quantity * price
        if code in self.balance:
            self.balance[code] = {
                'invested': self.balance[code]['quantity'] + quantity * price,
                'quantity': self.balance[code]['quantity'] + quantity
            }
        else:
            self.balance[code] = {
                'invested': quantity * price,
                'quantity': quantity
            }
        return quantity * price

    def sell(self, code, quantity):
        if quantity < 1:
            return 0

        dt = self.testing_date.strftime("%Y%m%d")
        price = self.universe[code]['price_df'].loc[dt, 'close']

        # add to deposit and update balance
        self.deposit += quantity * price

        self.balance[code] = {
            'invested': self.balance[code]['quantity'] - quantity * price,
            'quantity': self.balance[code]['quantity'] - quantity
        }

        return quantity * price

    def print_summary(self):
        """
        Sharpe (= avg annual Return / avg annual stddev of Return)
        Turnover (= Avg value traded / value held)
        Fitness (= Sharpe * sqrt(abs(Returns)) / max(Turnover, 0.125))
        Returns (= Avg profit or loss / invested amount * 2)
        Drawdown (= Largest reduction of profit and loss)
        Margin (= Avg profit or loss / amount traded)
        Cumulative Profit and Loss graph
        """

        # get data
        dt = np.array(list(self.summary.keys()))
        pnl = []
        traded = []
        held = []
        for s in self.summary.values():
            pnl.append(s['deposit'] + s['in stock'])
            traded.append(s['bought'] + s['sold'])
            held.append(s['in stock'])

        pnl = np.array(pnl)
        traded = np.array(traded)
        held = np.array(held)

        pnl = pnl / self.start_deposit

        # calculate summary data
        # Rate of return
        n = pnl.size
        period = (dt[n-1] - dt[0]).days
        ror = round((pow(pnl[n - 1], 365/period) - 1) * 100, 4)

        # Sharpe
        aror = []
        yrly_dt = dt[0]
        while yrly_dt < dt[n-1]:
            nxt_dt = yrly_dt + timedelta(days=365)
            i = find_nearest_idx(dt, yrly_dt)
            j = find_nearest_idx(dt, nxt_dt)

            aror.append((pow(pnl[j]/pnl[i], 365/(dt[j]-dt[i]).days) - 1) * 100)

            yrly_dt = nxt_dt

        sharpe = 0
        if np.std(aror) != 0:
            sharpe = round(np.mean(aror)/np.std(aror), 4)

        # Turnover
        turnover = round(np.mean(traded) / np.mean(held), 4)

        # Returns
        returns = round(ror / 2, 4)

        # Fitness
        fitness = round(sharpe * math.sqrt(abs(returns)) / max(turnover, 0.125), 4)

        # Drawdown
        pnl_d1 = []
        for i, p in enumerate(pnl):
            if i + 1 == pnl.size:
                continue
            pnl_d1.append(pnl[i] - pnl[i+1])

        pnl_d1 = np.array(pnl_d1)
        drawdown = round(np.max(pnl_d1) * 100, 4)

        # Margin
        margin = round(ror / (np.mean(traded) / self.start_deposit), 4)

        # matplotlib
        r = np.power(ror / 100 + 1, np.array([x.days for x in (dt - dt[0])]) / 365)

        plt.figure(figsize=(15, 4))
        txt = ('ROR: ' + str(ror) + '%' + '\n' +
               'Sharpe: ' + str(sharpe) + '\n' +
               'Turnover: ' + str(turnover) + '\n' +
               'Fitness: ' + str(fitness) + '\n' +
               'Returns: ' + str(returns) + '\n' +
               'Drawdown: ' + str(drawdown) + '%' + '\n' +
               'Margin: ' + str(margin) + '\n')

        print('________________Summary_________________')
        print(txt)
        plt.text(dt[int(round(n * 0.8))], pnl[int(round(n * 0.3))], txt, fontsize=7)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=300))
        plt.plot(dt, pnl, label='PnL')
        plt.plot(dt, r, label='Power regression')
        plt.gcf().autofmt_xdate()
        plt.show()
