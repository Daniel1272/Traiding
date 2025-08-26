import backtrader as bt

# 1️⃣ Кастомный DataFeed с твоими признаками
class MyPandasData(bt.feeds.PandasData):
    lines = tuple(
        [f"f{i}" for i in range(1, 9)] +
        [f"d{i}" for i in range(1, 8)] +
        [f"pct_{i}" for i in range(1, 7)] +
        [f"dirpct_{i}" for i in range(1, 7)]
    )

    params = (
        ('datetime', 'timestamp'),
        ('open', None),
        ('high', None),
        ('low', None),
        ('close', 'close'),
        ('volume', None),
        ('openinterest', None),
    ) + tuple((f"f{i}", f"f{i}") for i in range(1, 9)) \
      + tuple((f"d{i}", f"d{i}") for i in range(1, 8)) \
      + tuple((f"pct_{i}", f"pct_{i}") for i in range(1, 7)) \
      + tuple((f"dirpct_{i}", f"dirpct_{i}") for i in range(1, 7))


# 2️⃣ Стратегия, которая будет использовать ML модель
class MLStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('features', []),
        ('printlog', False)
    )

    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')

    def next(self):
        row = [getattr(self.datas[0], col)[0] for col in self.params.features]
        pred = self.params.model.predict([row])[0]
        if not self.position:
            if pred == 1:
                self.buy()
                self.log(f'BUY CREATE {self.datas[0].close[0]:.2f}')
            elif pred == 0:
                self.sell()
                self.log(f'SELL CREATE {self.datas[0].close[0]:.2f}')
        else:
            self.close()
            self.log(f'CLOSE {self.datas[0].close[0]:.2f}')

# 3️⃣ Функция запуска бэктеста
def run_backtest(df, model, features, cash=10000, printlog=False):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)

    data = MyPandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(MLStrategy, model=model, features=features, printlog=printlog)

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.run()
    print(f'Ending Portfolio Value: {cerebro.broker.getvalue():.2f}')

    cerebro.plot()
