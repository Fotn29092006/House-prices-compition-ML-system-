# House Prices — прогноз стоимости домов

Проект по машинному обучению: по характеристикам дома модель предсказывает
цену продажи. Данные взяты с Kaggle —
[Housing Prices Competition](https://www.kaggle.com/competitions/home-data-for-ml-course/data).

## Задача

Задача регрессии: на вход подаются признаки дома (площадь, год постройки,
район, качество отделки и т.д.), на выходе — цена `SalePrice` в долларах.

Качество модели в соревновании оценивается метрикой RMSE по логарифму цены:
`RMSE(log(prediction), log(actual))`. Логарифм нужен, чтобы ошибки на дорогих
и дешёвых домах весили одинаково.

## Данные

Используется датасет Ames Housing — продажи жилых домов в городе Эймс,
штат Айова, за 2006–2010 годы.

- `train.csv` — обучающая выборка, 1460 домов, 79 признаков + цена `SalePrice`
- `test.csv` — тестовая выборка, 1459 домов, те же признаки без цены
- `data_description.txt` — описание всех признаков
- `sample_submission.csv` — пример файла с ответами

Целевая переменная — `SalePrice` (цена продажи).

Признаки можно разбить на группы:

- площадь и размеры — `LotArea`, `GrLivArea`, `TotalBsmtSF`, `GarageArea`
- качество и состояние — `OverallQual`, `OverallCond`, `KitchenQual`, `ExterQual`
- годы — `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`
- комнаты — `BedroomAbvGr`, `FullBath`, `HalfBath`, `TotRmsAbvGrd`
- расположение — `Neighborhood`, `MSZoning`, `Condition1`
- тип дома — `BldgType`, `HouseStyle`, `MSSubClass`
- прочее — `GarageCars`, `Fireplaces`, `PoolArea`, `Fence`, `SaleCondition`

Признаки бывают числовые (площади, годы, количество), категориальные (район,
тип дома) и порядковые (оценки качества Ex/Gd/TA/Fa/Po). В данных есть
пропуски (`PoolQC`, `Alley`, `FireplaceQu` и др.), их нужно обработать.
Полное описание каждого поля — в `data/data_description.txt`.

## Структура проекта

```
house-prices-ml/
├── data/
│   ├── raw/            # исходные CSV с Kaggle (в git не хранятся)
│   └── processed/      # обработанные данные
├── notebooks/
│   ├── 01_eda.ipynb    # разведочный анализ
│   └── 02_modeling.ipynb
├── src/
│   ├── data_prep.py    # загрузка и очистка
│   ├── features.py     # признаки и препроцессинг
│   ├── train.py        # обучение
│   └── predict.py      # предсказание -> submission.csv
├── models/             # сохранённые модели
├── submissions/        # ответы для Kaggle
├── requirements.txt
└── README.md
```

## Запуск

```bash
git clone https://github.com/USERNAME/house-prices-ml.git
cd house-prices-ml

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# скачать данные с Kaggle и положить в data/raw/
python src/train.py             # обучение
python src/predict.py           # предсказание -> submissions/submission.csv
```

## План работы

1. EDA — распределения, пропуски, выбросы, корреляции с ценой
2. Препроцессинг — заполнение пропусков, кодирование категорий, масштабирование
3. Новые признаки — например `TotalSF`, возраст дома, общее число санузлов
4. Модели — линейная регрессия как базовая, затем Random Forest и градиентный бустинг
5. Валидация — кросс-валидация, RMSE на логарифме цены
6. Submission — собрать `submission.csv` и отправить на Kaggle

## Команда

Над проектом работают два человека.

- Участник 1 — EDA, препроцессинг, признаки (`src/data_prep.py`, `src/features.py`)
- Участник 2 — обучение моделей, валидация, submission (`src/train.py`, `src/predict.py`)

Работаем через ветки: каждый делает свою ветку под задачу и вливает изменения
в `main` через Pull Request.

## Стек

Python, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, Jupyter.
