# bt Strategy to Kiwoom

This repository accompanies the QuantTrading repository. It allows backtesting for your customized strategy

## bt Strategy

1. Select Universe
2. Weight (either from strategies or securities)
3. Allocate

## Kiwoom Strategy

1. Select Universe
2. Select Buy Condition
3. Select Sell Condition

## Strategy Equivalence
- **bt(1,2,3) ~ Kiwoom(1,2(if weight > 0),3(if weight < 0)**
- So Kiwoom Strategy is more general.

## Conversion Ideas
- ### Manual conversion (human conversion. Laborious and will take a lot of time, but works) (그냥 이거하기로)
- Modify the Kiwoom Strategy class to implement bt Strategy (Can be tricky. requires full understanding of the Kiwoom
API and bt) (probably impossible to do on my own)
- Abandon bt and make my own backtesting module. (Could be done, but also tricky.)

## Strategy Ideas 
- RSI
- Average True range
- Bollinger Band - time series operation
- Keltner channel
- ln(yesterday's close / yesterday's open)
- https://arxiv.org/ftp/arxiv/papers/1601/1601.00991.pdf

