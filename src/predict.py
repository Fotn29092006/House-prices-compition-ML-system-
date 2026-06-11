"""Предсказание цен для теста и сборка submission.csv."""
import joblib
import numpy as np
import pandas as pd

import data_prep
import features

MODEL_PATH = data_prep.ROOT / "models" / "model.pkl"
SUBMISSIONS_DIR = data_prep.ROOT / "submissions"


def main():
    pipe = joblib.load(MODEL_PATH)

    _, test = data_prep.load_raw()
    test = data_prep.fill_missing(test)
    test = features.add_features(test)

    ids = test[data_prep.ID]
    X_test = test.drop(columns=[data_prep.ID])
    X_test = X_test[data_prep.SELECTED_FEATURES]  # те же 15 признаков

    # модель обучена на log(цены) — возвращаем обратно через expm1
    pred = np.expm1(pipe.predict(X_test))
    pred = np.clip(pred, 0, None)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame({data_prep.ID: ids, data_prep.TARGET: pred})
    out_path = SUBMISSIONS_DIR / "submission.csv"
    out.to_csv(out_path, index=False)
    print(f"Сохранено {len(out)} предсказаний в {out_path}")


if __name__ == "__main__":
    main()
