import collections

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import talib
from db_helper import *
from const import *
from datetime import *
import talib


class Backtesting():

    def __init__(self):
        self.universe = {}
        self.universe_close = collections.defaultdict(dict)

        self.deposit = 10000000
        self.balance = {}

        # Adjust based on market sentiment.
        self.investment_deposit_ratio = 0.9  # [0, 1]
        self.investment_ratio = 0.5  # [0, 1]
        self.stop_loss = 0.06

        self.start_date = date(2023, 1, 1)  # .strftime('%Y%m%d')
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
            # to skip weekends
            if dt not in self.universe[sample_code]['price_df'].index:
                self.testing_date += timedelta(days=1)
                continue
            if dt not in self.universe_close:
                self.testing_date += timedelta(days=1)
                continue

            bought = 0
            sold = 0
            in_stock = 0
            total_quantity = 0
            daily_deposit = self.deposit
            print("Testing at: " + dt)

            # update weight for this date
            for code in codes:
                if dt not in self.universe[code]['price_df'].index:
                    continue
                self.alpha(code)

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

            for code in codes:
                if code not in self.balance:
                    continue
                if code not in self.universe_close[dt]:
                    continue
                # Calculate value in stock if it were to be sold today.
                total_quantity += self.balance[code]['quantity']
                in_stock += self.balance[code]['quantity'] * self.universe_close[dt][code]

            print("Deposit: " + str(self.deposit))
            print("In stock: " + str(in_stock))

            self.summary[self.testing_date] = {
                'valuation': self.deposit + in_stock,
                'bought': bought,
                'sold': sold
            }

            self.testing_date += timedelta(days=1)

        self.print_summary()

    def alpha(self, code):
        dt = self.testing_date.strftime("%Y%m%d")
        universe_item = self.universe[code]

        high = self.universe[code]['price_df']['high']
        low = self.universe[code]['price_df']['low']
        open = self.universe[code]['price_df']['open']
        close = self.universe[code]['price_df']['close']
        volume = self.universe[code]['price_df']['volume']

        # Bollinger Bands
        upper, mid, lower = talib.BBANDS(close,
                                         nbdevup=2,
                                         nbdevdn=2,
                                         timeperiod=20)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        # Stochastic (d is 3-day moving average of k)
        k, d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0,
                           slowd_period=3, slowd_matype=0)

        # Update weight
        self.weight[code] = macd_hist[dt]

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

        quantity = int(round(np.nan_to_num(daily_deposit / 200 / close * nw)))

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
        for s in self.summary.values():
            pnl.append(s['valuation'])
        pnl = np.array(pnl)
        pnl = pnl / 10000000

        plt.figure(figsize=(10, 4))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=100))
        plt.plot(dt, pnl)
        plt.gcf().autofmt_xdate()
        plt.show()


