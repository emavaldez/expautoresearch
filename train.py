"""Modelo del experimento. Es el ÚNICO archivo que modifica el loop de autoresearch.

Punto de partida: el modelo de clase del curso. Variables crudas (sin foto_mes), un mes de entrenamiento,
clase binaria BAJA+2 contra el resto, y los hiperparámetros del notebook de LightGBM de la materia.
La interfaz está documentada en prepare.py.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

import prepare

MESES_ENTRENAMIENTO = 1        # meses con clase (los más recientes que permite el gap)
HISTORIA_MESES = 0             # meses que make_features mira hacia atrás (lags)
SUBMUESTREO_CONTINUA = 1.0     # fracción de CONTINUA que se usa al entrenar (1.0 = todas)
NUM_ROUNDS = 1000

PARAMS = dict(
    objective="binary",
    learning_rate=0.027,
    num_leaves=8,
    min_data_in_leaf=76,
    feature_fraction=0.8,
    max_bin=31,
    verbosity=-1,
    deterministic=True,
    force_row_wise=True,
)

NO_SON_VARIABLES = ["foto_mes"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables crudas, tal cual vienen."""
    return df


def train_and_predict(train: pd.DataFrame, score: pd.DataFrame, seed: int, params: dict | None = None) -> np.ndarray:
    p = {**PARAMS, **(params or {})}
    rondas = int(p.pop("num_rounds", NUM_ROUNDS))
    fraccion = float(p.pop("submuestreo", SUBMUESTREO_CONTINUA))
    usar = prepare.muestra_continua(train, fraccion, seed)
    tr = train[usar]
    y = (tr[prepare.CLASE] == "BAJA+2").astype(int).to_numpy()
    columnas = [c for c in tr.columns if c not in NO_SON_VARIABLES + [prepare.CLASE]]
    modelo = lgb.train({**p, "seed": int(seed)}, lgb.Dataset(tr[columnas], label=y), num_boost_round=rondas)
    return modelo.predict(score[columnas])
