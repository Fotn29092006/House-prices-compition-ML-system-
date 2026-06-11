"""Обучение моделей и выбор лучшей по кросс-валидации."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import data_prep
import features

MODELS_DIR = data_prep.ROOT / "models"
RANDOM_STATE = 42


def get_models():
    """Кандидаты: линейная (базовая), случайный лес и градиентный бустинг."""
    return {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def main():
    train, _ = data_prep.load_raw()
    train = data_prep.fill_missing(train)
    train = features.add_features(train)

    # цену логарифмируем — метрика соревнования считается по log(цены)
    y = np.log1p(train[data_prep.TARGET])
    X = train.drop(columns=[data_prep.TARGET, data_prep.ID])
    X = X[data_prep.SELECTED_FEATURES]  # используем только 15 признаков
    print(f'Используется признаков: {X.shape[1]}')

    best_name, best_score, best_pipe = None, np.inf, None
    for name, model in get_models().items():
        pipe = Pipeline([("pre", features.build_preprocessor()), ("model", model)])
        scores = cross_val_score(
            pipe, X, y, scoring="neg_root_mean_squared_error", cv=5
        )
        rmse = -scores.mean()
        print(f"{name:<14} RMSE(log) = {rmse:.5f}")
        if rmse < best_score:
            best_name, best_score, best_pipe = name, rmse, pipe

    print(f"\nЛучшая модель: {best_name} (RMSE = {best_score:.5f})")

    # обучаем лучшую модель на всех данных и сохраняем
    best_pipe.fit(X, y)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODELS_DIR / "model.pkl")
    print(f"Модель сохранена в {MODELS_DIR / 'model.pkl'}")


if __name__ == "__main__":
    main()
