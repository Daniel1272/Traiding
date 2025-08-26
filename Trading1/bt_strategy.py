import backtrader as bt
import pandas as pd


class MyPandasData(bt.feeds.PandasData):
    lines = ('f1', 'f2', 'f3', 'd1', 'd2', 'd3', 'dirpct_1', 'dirpct_2', 'dirpct_3')
    params = (
        ('datetime', 'timestamp'),
        ('open', None),
        ('high', None),
        ('low', None),
        ('close', 'close'),
        ('volume', None),
        ('openinterest', None),
    )
class ModelStrategy(bt.Strategy):
    params = (
        ('model', None),       # обученная модель
        ('features', []),      # список колонок признаков для модели
        ('printlog', False),
    )

    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')

    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        # подготавливаем данные для модели
        row = pd.DataFrame([{col: getattr(self.datas[0], col)[0] for col in self.params.features}])
        signal = self.params.model.predict(row)[0]

        if not self.position:
            if signal == 1:
                self.buy()
                self.log(f'BUY CREATE {self.dataclose[0]:.2f}')
            elif signal == 0:
                self.sell()
                self.log(f'SELL CREATE {self.dataclose[0]:.2f}')
        else:
            # закрываем позицию на следующей свече
            self.close()
            self.log(f'CLOSE {self.dataclose[0]:.2f}')

def run_backtest(df, model, features, cash=10000, printlog=False):
    """
    df: DataFrame с колонками ['timestamp', 'close', f-признаки...]
    model: обученная модель sklearn
    features: список признаков для модели
    cash: стартовый капитал
    printlog: вывод логов
    """
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)

    # создаём DataFeed
    data = MyPandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(ModelStrategy, model=model, features=features, printlog=printlog)

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.run()
    print(f'Ending Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.plot()



