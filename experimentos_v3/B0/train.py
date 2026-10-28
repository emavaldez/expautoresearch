"""v3/train.py: el ÚNICO archivo que edita el loop (ver PROTOCOLO.md y v3/program.md).

Arranca como el baseline de la cátedra (B0):
  - variables: las 152 crudas, más lag 1 y 2 y delta 1 y 2 de cada una;
  - clase binaria: BAJA+2 contra el resto (CONTINUA y BAJA+1);
  - LightGBM binario, entrenado con 201903 a 201906 completos, sin submuestreo;
  - hiperparámetros por búsqueda (v3/hpo.py, 40 trials ordenados por AUC en 201908), rondas por early stopping en 201908.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

ID, MES, CLASE = "numero_de_cliente", "foto_mes", "clase_ternaria"

MESES_ENTRENAMIENTO = [201903, 201904, 201905, 201906]

# El espacio de la cátedra. bagging_freq va fijo en 1: sin él, bagging_fraction no hace nada.
ESPACIO = {
    "num_leaves": ("intlog", 8, 512),
    "learning_rate": ("log", 0.01, 0.2),
    "min_data_in_leaf": ("intlog", 20, 5000),
    "feature_fraction": ("float", 0.1, 1.0),
    "bagging_fraction": ("float", 0.3, 1.0),
}
FIJOS = {"objective": "binary", "metric": "auc", "bagging_freq": 1, "verbose": -1, "feature_pre_filter": False}
RONDAS_MAX, PACIENCIA = 3000, 100

# Lo reemplaza el mejor trial de la búsqueda del baseline (experimentos_v3/B0/hpo.json).
PARAMS = {"num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 100, "feature_fraction": 0.5,
          "bagging_fraction": 0.8}

LAGS = (1, 2)


# ---------------------------------------------------------------- features

def lags_y_deltas(df: pd.DataFrame, lags=LAGS) -> pd.DataFrame:
    """Lag k: el valor del cliente k filas antes (meses consecutivos); delta k: actual menos lag k."""
    base = [c for c in df.columns if c not in (ID, MES)]
    orden = np.lexsort((df[MES].to_numpy(), df[ID].to_numpy()))
    ids = df[ID].to_numpy()[orden]
    v = df[base].to_numpy(dtype=np.float32)[orden]
    nuevas = {}
    for k in lags:
        mismo = np.zeros(len(ids), dtype=bool)
        mismo[k:] = ids[k:] == ids[:-k]
        lag = np.full_like(v, np.nan)
        lag[k:] = v[:-k]
        lag[~mismo] = np.nan
        lag_o, delta_o = np.empty_like(lag), np.empty_like(lag)
        lag_o[orden], delta_o[orden] = lag, v - lag  # de vuelta al orden de las filas de entrada
        nuevas.update({f"{c}_lag{k}": lag_o[:, j] for j, c in enumerate(base)})
        nuevas.update({f"{c}_delta{k}": delta_o[:, j] for j, c in enumerate(base)})
    return pd.concat([df, pd.DataFrame(nuevas, index=df.index)], axis=1)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    return lags_y_deltas(df)


def columnas(df: pd.DataFrame) -> list[str]:
    """Todo menos el cliente, el mes (identificaría el mes) y la clase."""
    return [c for c in df.columns if c not in (ID, MES, CLASE)]


# ---------------------------------------------------------------- modelo

_DATASETS: dict = {}  # la búsqueda llama muchas veces con los mismos datos: el binning se hace una vez


def _datasets(tr, va, y_va, cols, y):
    clave = (id(tr), id(va), tuple(cols))
    if clave not in _DATASETS:
        _DATASETS.clear()
        dtr = lgb.Dataset(tr[cols].to_numpy(np.float32), y, feature_name=cols, free_raw_data=False,
                          params={"feature_pre_filter": False, "verbose": -1})
        dva = lgb.Dataset(va[cols].to_numpy(np.float32), y_va.astype(int), reference=dtr, free_raw_data=False)
        _DATASETS[clave] = (dtr, dva)
    return _DATASETS[clave]


def objetivo(tr: pd.DataFrame) -> np.ndarray:
    return (tr[CLASE] == "BAJA+2").to_numpy().astype(int)


def train_and_predict(tr, va, y_va, te, params, seed):
    cols = columnas(tr)
    dtr, dva = _datasets(tr, va, y_va, cols, objetivo(tr))
    p = {**FIJOS, **params, "seed": int(seed)}
    m = lgb.train(p, dtr, RONDAS_MAX, valid_sets=[dva], callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)])
    p_va = m.predict(va[cols].to_numpy(np.float32), num_iteration=m.best_iteration)
    p_te = None if te is None else m.predict(te[cols].to_numpy(np.float32), num_iteration=m.best_iteration)
    return p_va, p_te, {"rondas": int(m.best_iteration), "variables": len(cols)}
