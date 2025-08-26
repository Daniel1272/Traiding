import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.options.display.expand_frame_repr = False
# создаём подключение к бирже (без ключей можно тянуть только публичные данные)
exchange = ccxt.binance()

# загружаем исторические данные (OHLCV = Open, High, Low, Close, Volume)
symbol = 'BTC/USDT'
timeframe = '1h'   # можно '1m', '5m', '1h', '1d'
limit = 10000       # максимум за один запрос

ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# переводим в DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df = df.copy()[['timestamp', 'volume', 'close']]

# преобразуем время
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

def find_pivot_points(df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
    """
    Находит локальные точки разворота (экстремумы) по выбранной колонке (по умолчанию 'close').
    Возвращает DataFrame с колонкой 'pivot':
      1  -> локальный максимум
     -1  -> локальный минимум
      0  -> нет сигнала
    """
    df['pivot'] = 0
    for i in range(1, len(df) - 1):
        if df[column].iloc[i] > df[column].iloc[i - 1] and df[column].iloc[i] > df[column].iloc[i + 1]:
            df.loc[df.index[i], 'pivot'] = 1   # максимум
        elif df[column].iloc[i] < df[column].iloc[i - 1] and df[column].iloc[i] < df[column].iloc[i + 1]:
            df.loc[df.index[i], 'pivot'] = -1  # минимум
    return df


def add_waves_from_pivots(df: pd.DataFrame, column: str = 'close', pivot_col: str = 'pivot',
                          num_waves: int = 8) -> pd.DataFrame:
    """
    Добавляет колонки wave_1 ... wave_num_waves, которые отражают последовательные волны
    (разницу между соседними экстремумами: максимум <-> минимум).

    column: цена (обычно 'close')
    pivot_col: колонка с pivot (1 = максимум, -1 = минимум, 0 = нет сигнала)
    """
    # индексы pivot-точек
    pivots = df.index[df[pivot_col] != 0].tolist()

    # создаём пустые колонки
    for n in range(1, num_waves + 1):
        df[f'wave_{n}'] = None

    # обходим все pivot-точки
    for i, idx in enumerate(pivots):
        price_now = df.loc[idx, column]

        # смотрим назад на num_waves шагов
        for n in range(1, num_waves + 1):
            if i - n >= 0:
                prev_idx = pivots[i - n]
                price_prev = df.loc[prev_idx, column]
                # волна = разница между текущим pivot и предыдущим
                df.loc[idx, f'wave_{n}'] = price_now - price_prev

    return df



def build_wave_features_diffs_pct_dir(
    df: pd.DataFrame,
    pivot_col: str = 'pivot',
    wave_col: str = 'wave_1',
    num_features: int = 8
) -> pd.DataFrame:
    """
    Для каждой pivot-точки формирует:
      - f1..fN : последние N волн (со знаком)
      - d1..d(N-1) : абсолютные расстояния между соседними f (abs)
      - pct_1..pct_(N-2) : симметричное процентное соотношение между d_n и d_{n+1}
                          pct = min(d_n,d_{n+1})/max(d_n,d_{n+1})*100  (0..100)
      - dirpct_1..dirpct_(N-2) : направленное соотношение d_n/d_{n+1}*100
                          (если d_{n+1} == 0 and d_n == 0 => 100.0;
                           если d_{n+1} == 0 and d_n != 0 => NaN)
    Возвращает DataFrame с отброшенными NaN-строками (недостаток истории).
    """
    pivots = df[df[pivot_col] != 0].copy().reset_index(drop=True)

    # f1..fN
    for n in range(1, num_features + 1):
        pivots[f'f{n}'] = pivots[wave_col].shift(n - 1)

    # d1..d(N-1) = abs differences between neighbouring f
    for n in range(1, num_features):
        pivots[f'd{n}'] = (pivots[f'f{n}'] - pivots[f'f{n+1}']).abs()

    # pct_1..pct_(N-2) = symmetric percent similarity between d_n and d_{n+1}
    for n in range(1, num_features - 1):
        a = pivots[f'd{n}'].to_numpy(dtype=float)
        b = pivots[f'd{n+1}'].to_numpy(dtype=float)
        max_ab = np.maximum(a, b)
        min_ab = np.minimum(a, b)
        # if both zero -> 100.0; if max==0 but min!=0 (shouldn't happen) -> nan; else min/max*100
        pct = np.where(max_ab == 0, np.where(min_ab == 0, 100.0, np.nan), (min_ab / max_ab) * 100.0)
        pivots[f'pct_{n}'] = pct

    # dirpct_1..dirpct_(N-2) = directional percent = d_n / d_{n+1} * 100
    for n in range(1, num_features - 1):
        a = pivots[f'd{n}'].to_numpy(dtype=float)
        b = pivots[f'd{n+1}'].to_numpy(dtype=float)
        # if both zero -> 100.0 (equal); if denom zero and numer !=0 -> NaN; else a/b*100
        dirpct = np.where(
            (b == 0) & (a == 0), 100.0,
            np.where(b == 0, np.nan, (a / b) * 100.0)
        )
        pivots[f'dirpct_{n}'] = dirpct

    pivots = pivots.dropna().reset_index(drop=True)
    return pivots


# пример использования:
df = find_pivot_points(df, column='close')
df = add_waves_from_pivots(df, column='close', pivot_col='pivot', num_waves=1)
features =  build_wave_features_diffs_pct_dir(df, pivot_col='pivot', wave_col='wave_1', num_features=8)
print(features.columns)


