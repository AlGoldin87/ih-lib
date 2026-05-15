# ih-lib: Information-Theoretic Analysis Core

`ih-lib` — это вычислительное ядро IH-анализа. Библиотека предоставляет быстрые реализации на C++ с Python-интерфейсом для измерения связей в данных и построения решающих правил.

## Что такое IH-анализ

Это метод, который позволяет измерить степень взаимосвязи между признаками и целевой переменной по одной универсальной формуле, работающей одинаково для чисел, категорий и их комбинаций. Узнать больше о методе можно в [статье на Хабре](https://habr.com/ru/articles/XXXXXX/).

## Установка

    pip install git+https://github.com/AlGoldin87/ih-lib.git

## Основные функции

### `calculate_entropy(data, mask)` — Измерение связей

Вычисление энтропии Шеннона для любой комбинации признаков в подготовленной матрице имён. Это фундамент, на основе которого строится ключевая метрика метода **R(Y|X)** — степень детерминированности связи между признаками X и целевой переменной Y.

    import pandas as pd
    import numpy as np
    from ih_prep import prepare_data
    from ih import calculate_entropy

    # Загрузите и подготовьте данные
    df = pd.read_csv('your_data.csv')
    data, info = prepare_data(df, target='target_column', sharpness=0.25)

    # Вычислите R(Y|X) для группы признаков [0, 1]
    target_mask = np.zeros(data.shape[1], dtype=np.int32)
    target_mask[-1] = 1
    Hy = calculate_entropy(data, target_mask)

    mask_x = np.zeros(data.shape[1], dtype=np.int32)
    mask_x[0] = mask_x[1] = 1
    Hx = calculate_entropy(data, mask_x)

    mask_xy = mask_x | target_mask
    Hxy = calculate_entropy(data, mask_xy)

    Ixy = Hx + Hy - Hxy
    Rxy = Ixy / Hy
    print(f"R(Y|X) = {Rxy:.4f}")  # 0 — связи нет, 1 — функциональная связь

### `find_best_rules(...)` — Построение правил

Поиск оптимальных бинарных правил «ЕСЛИ-ТО» для задачи классификации. Функция находит пороги для количественных и подмножества для категориальных признаков, максимизируя R(Y|X). *(Тема следующих статей.)*

    from ih import find_best_rules

    results = find_best_rules(
        prepared_data=data,
        feature_mask=feature_mask,   # 1-колич., 2-категор.
        y_index=target_column_index,
        feature_names=feature_names
    )

    for r in results:
        print(f"{r['feature_name']}: R(Y|X)={r['Rxy']:.4f}")

## Производительность

- **C++ backend** для быстрых вычислений
- **Python API** для интеграции со стандартным стеком (pandas, numpy)
- Точность энтропийных расчётов проверена на эталонных значениях

## Экосистема IH-анализа

- **[`ih-prep`](https://github.com/AlGoldin87/ih-prep)** — подготовка данных: сырой DataFrame → матрица имён
- **[`ih-coverage`](https://github.com/AlGoldin87/ih-coverage)** — автоматический подбор оптимальной резкости (ICC)
