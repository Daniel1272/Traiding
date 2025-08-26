import ccxt
import pandas as pd
import joblib
from BTC_preprocessing import find_pivot_points,add_waves_from_pivots,build_wave_features_diffs_pct_dir
from bt_strategy import run_backtest


exchange = ccxt.binance()

# загружаем исторические данные (OHLCV = Open, High, Low, Close, Volume)
symbol = 'BTC/USDT'
timeframe = '5m'   # можно '1m', '5m', '1h', '1d'
limit = 10000       # максимум за один запрос

ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# переводим в DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.copy()[['timestamp', 'volume', 'close']]
# загружаем модель
model = joblib.load('rf_model.pkl')


# пример использования:
df = find_pivot_points(df, column='close')
df = add_waves_from_pivots(df, column='close', pivot_col='pivot', num_waves=1)
features =  build_wave_features_diffs_pct_dir(df, pivot_col='pivot', wave_col='wave_1', num_features=8)
# используем признаки для модели
features_list = [col for col in features.columns if col.startswith('f')]

# запускаем бэктест
run_backtest(features, model=model, features=features_list, cash=10000, printlog=True)
