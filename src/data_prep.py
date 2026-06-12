"""Загрузка исходных данных и базовая очистка."""
from pathlib import Path

import pandas as pd

# пути к данным
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# датасет ищем в нескольких местах: сначала data/raw, затем папка,
# которую скачивают напрямую с Kaggle (home-data-for-ml-course)
_CANDIDATE_DIRS = [
    ROOT / "data" / "raw",
    ROOT / "home-data-for-ml-course",
    ROOT,
]


def _find_raw_dir():
    for d in _CANDIDATE_DIRS:
        if (d / "train.csv").exists():
            return d
    raise FileNotFoundError(
        "Не нашёл train.csv. Положите train.csv и test.csv в папку data/raw/ "
        "или home-data-for-ml-course/."
    )


RAW_DIR = _find_raw_dir()

TARGET = "SalePrice"
ID = "Id"

# для этих признаков пропуск означает "такого элемента в доме нет"
# (нет подвала, гаража, бассейна...), а не отсутствие данных - ставим "None"
NONE_COLS = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
    "MiscFeature",
]

# здесь пропуск логично заменить на 0 (нет гаража -> 0 машиномест и т.д.)
ZERO_COLS = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea", "GarageYrBlt",
]


def load_raw():
    """Читает train.csv и test.csv из найденной папки с данными."""
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    return train, test


def fill_missing(df):
    """Заполняет пропуски там, где они имеют понятный смысл."""
    df = df.copy()
    for col in NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    for col in ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    # MSSubClass закодирован числом, но по смыслу это тип дома (категория)
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)
    return df


# 15 признаков, отобранных по связи с ценой (см. ноутбук EDA):
SELECTED_FEATURES = [
    "OverallQual",    # общее качество дома (1-10)
    "GrLivArea",      # жилая площадь
    "GarageCars",     # мест в гараже
    "GarageArea",     # площадь гаража
    "TotalBsmtSF",    # площадь подвала
    "1stFlrSF",       # площадь первого этажа
    "FullBath",       # полноценные санузлы
    "TotRmsAbvGrd",   # всего комнат
    "YearBuilt",      # год постройки
    "YearRemodAdd",   # год последнего ремонта
    "Fireplaces",     # камины
    "LotArea",        # площадь участка
    "Neighborhood",   # район (категориальный)
    "KitchenQual",    # качество кухни
    "ExterQual",      # качество внешней отделки
]
