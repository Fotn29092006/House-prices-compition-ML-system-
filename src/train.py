"""Обучение моделей и выбор лучшей по кросс-валидации."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
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

    # цену логарифмируем - метрика соревнования считается по log(цены)
    y = np.log1p(train[data_prep.TARGET])
    X = train.drop(columns=[data_prep.TARGET, data_prep.ID])
    X = X[data_prep.SELECTED_FEATURES]  # используем только 15 признаков
    print(f'Используется признаков: {X.shape[1]}')

    # три базовые метрики оценки моделей:
    # RMSE и MAE - средняя ошибка (меньше = лучше), R2 - доля объяснённой
    # дисперсии от 0 до 1 (больше = лучше)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    print(f"\n{'модель':<14}{'RMSE':>9}{'MAE':>9}{'R2':>8}")
    best_name, best_score, best_pipe = None, np.inf, None
    for name, model in get_models().items():
        pipe = Pipeline([("pre", features.build_preprocessor()), ("model", model)])
        cv = cross_validate(pipe, X, y, scoring=scoring, cv=5)
        rmse = -cv["test_rmse"].mean()
        mae = -cv["test_mae"].mean()
        r2 = cv["test_r2"].mean()
        print(f"{name:<14}{rmse:>9.4f}{mae:>9.4f}{r2:>8.3f}")
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
