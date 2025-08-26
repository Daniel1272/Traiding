import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from BTC_preprocessing import features
import matplotlib.pyplot as plt
import joblib


pivots = features
# создаём таргет: 1 = следующая f1 > 0, 0 = следующая f1 <= 0
pivots['f1_dir'] = (pivots['f1'].shift(-1) > 0).astype(int)

# последняя строка будет NaN в таргете → её уберём
pivots = pivots.dropna(subset=['f1_dir']).reset_index(drop=True)

def walk_forward_validation_verbose(df, features, target, train_size=0.7, step_size=10):
    n = len(df)
    train_end = int(n * train_size)
    predictions = []
    y_true = []
    accuracies = []
    steps = []

    while train_end < n:
        X_train = df[features].iloc[:train_end]
        y_train = df[target].iloc[:train_end]
        X_test = df[features].iloc[train_end:train_end + step_size]
        y_test = df[target].iloc[train_end:train_end + step_size]

        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # сохраняем результаты
        predictions.extend(y_pred)
        y_true.extend(y_test.tolist())

        # точность текущего шага
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        steps.append(train_end)
        print(f"Step {train_end}: Accuracy = {acc:.3f}")

        train_end += step_size

    # общий отчёт
    print("\nOverall Accuracy:", accuracy_score(y_true, predictions))
    print("\nClassification Report:\n", classification_report(y_true, predictions))

    # график динамики точности
    plt.figure(figsize=(10,4))
    plt.plot(steps, accuracies, marker='o')
    plt.xlabel("Train end index")
    plt.ylabel("Step Accuracy")
    plt.title("Walk-forward step accuracy")
    plt.grid(True)
    plt.show()

    return y_true, predictions, accuracies



features = [col for col in pivots.columns if col.startswith(('f','d','pct','dirpct'))]
y_true, y_pred, accuracies = walk_forward_validation_verbose(
    pivots, features, target='f1_dir', step_size=10)

X = pivots[features]
y = pivots['f1_dir']

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X, y)

# сохраняем модель на диск
joblib.dump(model, 'rf_model.pkl')

