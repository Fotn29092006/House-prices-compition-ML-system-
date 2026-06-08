"""Загрузка исходных данных и базовая очистка."""
from pathlib import Path

import pandas as pd

# пути к данным
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "SalePrice"
ID = "Id"

# для этих признаков пропуск означает "такого элемента в доме нет"
# (нет подвала, гаража, бассейна...), а не отсутствие данных — ставим "None"
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
    """Читает train.csv и test.csv из data/raw."""
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
