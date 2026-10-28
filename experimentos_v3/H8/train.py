"""v3/train.py: el ÚNICO archivo que edita el loop (ver PROTOCOLO.md y v3/program.md).

Con CAMBIOS todo en False es el baseline de la cátedra (B0):
  - variables: las 152 crudas, más lag 1 y 2 y delta 1 y 2 de cada una;
  - clase binaria: BAJA+2 contra el resto (CONTINUA y BAJA+1);
  - LightGBM binario, entrenado con 201903 a 201906 completos, sin submuestreo;
  - hiperparámetros por búsqueda (v3/correr.py --busqueda), rondas por early stopping en 201908.

Cada idea de la cola (hipotesis.tsv, grupo loop) es una entrada de CAMBIOS. El loop prende una por paso: si queda, el
archivo con ese interruptor prendido pasa a ser el vigente (y se commitea); si no, se descarta.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ID, MES, CLASE = "numero_de_cliente", "foto_mes", "clase_ternaria"

CAMBIOS = {
    "H2": False,    # razones y ventana de 3 meses (montos y conteos)
    "H1b": True,   # BAJA+1 también como positivo
    "H6": False,    # entrenar con 201901 a 201906
    "H20": False,   # BAJA+1 a medio peso
    "H21": False,   # tendencia y promedio de 3 meses de la cátedra
    "H12": False,   # lags y deltas de 1 a 6
    "H1": False,    # multiclase
    "H3": False,    # todos los montos a percentil por mes
    "H22": False,   # percentil sólo en los montos que derivan
    "H27": False,   # ensamble con XGBoost
    "H4": False,    # variables rotas a NA
    "H7": False,    # especialista internet = 0
    "H24": None,    # especialista con otro segmento: "internet0", "ctrx" o "tarjeta"
    "H8": True,    # segunda etapa en la franja dudosa
    "H16": False,   # fultimo_cierre menos la moda del mes
    "H17": False,   # sin historia de Visa_ y Master_
    "H14": False,   # pesos por costo
    "H30": False,   # TabPFN en la franja (necesita instalar tabpfn)
    "H19": False,   # internet sólo del último mes de entrenamiento
    "H32": False,   # lags por mes calendario
    "H13": False,   # canaritos
    "H31": False,   # selección por SHAP
    "H5": False,    # submuestreo de CONTINUA al 20%
    "H23": False,   # sólo 201904 a 201906
    "H26": False,   # BAJA+2 sintéticas (SMOTE)
    "H29": False,   # red sobre la historia (necesita instalar torch)
}

MESES_ENTRENAMIENTO = ([201904, 201905, 201906] if CAMBIOS["H23"] else
                       [201901, 201902, 201903, 201904, 201905, 201906] if CAMBIOS["H6"] else
                       [201903, 201904, 201905, 201906])
LAGS = tuple(range(1, 7)) if CAMBIOS["H12"] else (1, 2)

# El espacio de la cátedra. bagging_freq va fijo en 1: sin él, bagging_fraction no hace nada.
ESPACIO = {
    "num_leaves": ("intlog", 8, 512),
    "learning_rate": ("log", 0.01, 0.2),
    "min_data_in_leaf": ("intlog", 20, 5000),
    "feature_fraction": ("float", 0.1, 1.0),
    "bagging_fraction": ("float", 0.3, 1.0),
}
# H10: el espacio ampliado por el agente (mismo presupuesto de trials).
ESPACIO_AMPLIADO = {
    **ESPACIO,
    "num_leaves": ("intlog", 8, 2048),
    "lambda_l1": ("log", 1e-3, 100.0),
    "lambda_l2": ("log", 1e-3, 100.0),
    "min_gain_to_split": ("float", 0.0, 1.0),
    "max_depth": ("cat", [-1, 6, 8, 12, 16]),
    "path_smooth": ("log", 1e-3, 100.0),
    "extra_trees": ("cat", [False, True]),
}
FIJOS = {"bagging_freq": 1, "verbose": -1, "feature_pre_filter": False}
RONDAS_MAX, PACIENCIA = 3000, 100

# El mejor trial de la búsqueda del baseline (experimentos_v3/B0/hpo.json, trial 34: AUC 0,9116 en 201908).
PARAMS = {"num_leaves": 56, "learning_rate": 0.026251230543003472, "min_data_in_leaf": 4997,
          "feature_fraction": 0.6594210941714631, "bagging_fraction": 0.7742347538844919}

TARJETAS = ("Visa_", "Master_")
# Empiezan con c pero no son conteos: edades (crecen solas) y banderas 0/1.
NO_MAGNITUD = {"cliente_edad", "cliente_antiguedad", "cliente_vip", "cdescubierto_preacordado", "ccajas_transacciones",
               "cmobile_app_trx"}


def es_magnitud(c: str) -> bool:
    """Montos (m*) y conteos (c*), del cliente o de sus tarjetas. Afuera fechas, banderas y códigos."""
    nombre = c.split("_", 1)[1] if c.startswith(TARJETAS) else c
    return nombre[0] in "mc" and c not in NO_MAGNITUD


def es_monto(c: str) -> bool:
    nombre = c.split("_", 1)[1] if c.startswith(TARJETAS) else c
    return nombre[0] == "m"


def base_cols(df) -> list[str]:
    return [c for c in df.columns if c not in (ID, MES, CLASE)]


def razon(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0, np.nan, num / den).astype(np.float32)


# ---------------------------------------------------------------- tratamientos antes de la historia

def rotas_a_na(df: pd.DataFrame) -> None:
    """H4. Una variable está rota en un mes si es casi toda cero o vacía (≥ 95%) cuando en los meses anteriores no lo
    era (mediana ≤ 50%). Sólo mira el mes y los anteriores. Se pasa a NA en ese mes, antes de lags y deltas."""
    meses = sorted(df[MES].unique())
    cols = base_cols(df)
    mes = df[MES].to_numpy()
    z = np.array([((v := df.loc[mes == m, cols].to_numpy(np.float32)) == 0).mean(axis=0) + np.isnan(v).mean(axis=0)
                  for m in meses])
    rotas = []
    for i in range(1, len(meses)):
        antes = np.median(z[max(0, i - 3):i], axis=0)
        for j in np.flatnonzero((z[i] >= 0.95) & (antes <= 0.5)):
            v = df[cols[j]].to_numpy(np.float32, copy=True)
            v[mes == meses[i]] = np.nan
            df[cols[j]] = v
            rotas.append(f"{cols[j]}@{meses[i]}")
    print(f"  rotas a NA: {', '.join(rotas) or 'ninguna'}", flush=True)


def _percentil(v: np.ndarray) -> np.ndarray:
    """Percentil con cero fijo: los positivos entre sí (0, 100], los negativos entre sí [-100, 0), el cero en 0."""
    out = np.full(len(v), np.nan, dtype=np.float32)
    ok = ~np.isnan(v)
    out[ok & (v == 0)] = 0
    for s in (1, -1):
        k = ok & (s * v > 0)
        if k.any():
            out[k] = s * 100 * rankdata(s * v[k]) / k.sum()
    return out


def a_percentil(df: pd.DataFrame, cols: list[str]) -> None:
    """Percentil dentro de cada mes: sólo usa las filas de ese mes, no mira ni adelante ni atrás."""
    mes = df[MES].to_numpy()
    grupos = [mes == m for m in np.unique(mes)]
    for c in cols:
        v = df[c].to_numpy(np.float32)
        out = np.empty(len(v), dtype=np.float32)
        for f in grupos:
            out[f] = _percentil(v[f])
        df[c] = out


def montos_que_derivan(df: pd.DataFrame) -> list[str]:
    """H22 (regla de la primera): montos con algún cuartil de sus valores no nulos de 201907-201908 a más de 25% del de
    los meses de entrenamiento. Mira las variables de validación, nunca la clase ni 201910."""
    viejo = df[MES].isin(MESES_ENTRENAMIENTO).to_numpy()
    nuevo = df[MES].isin([201907, 201908]).to_numpy()
    out = []
    for c in (c for c in base_cols(df) if es_monto(c)):
        a, b = df.loc[viejo, c].to_numpy(np.float64), df.loc[nuevo, c].to_numpy(np.float64)
        a, b = a[(a != 0) & ~np.isnan(a)], b[(b != 0) & ~np.isnan(b)]
        if len(a) < 100 or len(b) < 100:
            continue
        qa, qb = np.quantile(a, [0.25, 0.5, 0.75]), np.quantile(b, [0.25, 0.5, 0.75])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(qb / qa - 1)
        if np.any(r[qa != 0] > 0.25):
            out.append(c)
    print(f"  montos que derivan ({len(out)}): {', '.join(out)}", flush=True)
    return out


def restar_moda(df: pd.DataFrame) -> None:
    """H16: días desde el último cierre menos la moda del mes (queda el grupo de cierre del cliente)."""
    for c in ("Visa_fultimo_cierre", "Master_fultimo_cierre"):
        df[c] = df[c] - df.groupby(MES)[c].transform(lambda v: v.mode().iloc[0] if v.notna().any() else 0)


# ---------------------------------------------------------------- historia

def _periodo(mes):
    return (mes // 100) * 12 + mes % 100


def lags_y_deltas(df: pd.DataFrame, lags=LAGS, sin_historia=(), calendario=False) -> pd.DataFrame:
    """Lag k: el valor del cliente k meses antes; delta k: actual menos lag k. Por fila (meses consecutivos del
    cliente) o, con calendario=True (H32), por mes calendario (NA si ese mes no estaba)."""
    base = [c for c in base_cols(df) if not c.startswith(sin_historia)]
    ids, per = df[ID].to_numpy(np.int64), _periodo(df[MES].to_numpy(np.int64))
    v = df[base].to_numpy(dtype=np.float32)
    nuevas = {}
    if calendario:
        pos = pd.Series(np.arange(len(df)), index=ids * 100_000 + per)
    else:
        orden = np.lexsort((per, ids))
        ids_o, v_o = ids[orden], v[orden]
    for k in lags:
        if calendario:
            j = pos.reindex(ids * 100_000 + per - k).to_numpy()
            ok = ~np.isnan(j)
            lag = np.full_like(v, np.nan)
            lag[ok] = v[j[ok].astype(np.int64)]
        else:
            mismo = np.zeros(len(ids_o), dtype=bool)
            mismo[k:] = ids_o[k:] == ids_o[:-k]
            lag_o = np.full_like(v_o, np.nan)
            lag_o[k:] = v_o[:-k]
            lag_o[~mismo] = np.nan
            lag = np.empty_like(lag_o)
            lag[orden] = lag_o  # de vuelta al orden de las filas de entrada
        delta = v - lag
        nuevas.update({f"{c}_lag{k}": lag[:, j_] for j_, c in enumerate(base)})
        nuevas.update({f"{c}_delta{k}": delta[:, j_] for j_, c in enumerate(base)})
    return pd.concat([df, pd.DataFrame(nuevas, index=df.index)], axis=1)


def razones_y_ventana(X: pd.DataFrame) -> dict:
    """H2 (primera competencia): sobre el mes actual, lag1 y lag2 de cada monto o conteo."""
    out = {}
    for c in (c for c in base_cols(X) if es_magnitud(c) and f"{c}_lag2" in X.columns):
        x, l1, l2 = (X[n].to_numpy(np.float32) for n in (c, f"{c}_lag1", f"{c}_lag2"))
        v = np.stack([x, l1, l2])
        n = (~np.isnan(v)).sum(axis=0)
        prom = razon(np.nansum(v, axis=0).round(2), n)
        with np.errstate(all="ignore"):
            maxi, mini = np.fmax.reduce(v), np.fmin.reduce(v)
        out.update({f"{c}_ratiolag1": razon(x, l1), f"{c}_ratioavg3": razon(x, prom), f"{c}_avg3": prom,
                    f"{c}_min3": mini, f"{c}_max3": maxi, f"{c}_ratiomax3": razon(x, maxi)})
    return out


def tendencia_y_promedio(X: pd.DataFrame) -> dict:
    """H21 (FE histórico de la cátedra): pendiente de los últimos 3 valores y su promedio. Sin las tarjetas ni internet,
    como en la primera; el promedio, sólo donde H2 no lo puso."""
    out = {}
    for c in base_cols(X):
        if c.startswith(TARJETAS) or c.startswith("internet") or f"{c}_lag1" not in X.columns or "_lag" in c \
                or "_delta" in c or c.endswith(("_ratiolag1", "_ratioavg3", "_avg3", "_min3", "_max3", "_ratiomax3")):
            continue
        x, l1 = X[c].to_numpy(np.float32), X[f"{c}_lag1"].to_numpy(np.float32)
        l2 = X[f"{c}_lag2"].to_numpy(np.float32)
        out[f"{c}_tend3"] = np.where(~np.isnan(l2), (x - l2) / 2, x - l1).astype(np.float32)
        if f"{c}_avg3" not in X.columns:
            v = np.stack([x, l1, l2])
            out[f"{c}_avg3p"] = razon(np.nansum(v, axis=0), (~np.isnan(v)).sum(axis=0))
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    if CAMBIOS["H4"]:
        rotas_a_na(df)
    if CAMBIOS["H3"]:
        a_percentil(df, [c for c in base_cols(df) if es_monto(c)])
    elif CAMBIOS["H22"]:
        a_percentil(df, montos_que_derivan(df))
    if CAMBIOS["H16"]:
        restar_moda(df)
    seq = historia_red(df) if CAMBIOS["H29"] else {}
    X = lags_y_deltas(df, LAGS, TARJETAS if CAMBIOS["H17"] else (), CAMBIOS["H32"])
    if seq:
        X = pd.concat([X, pd.DataFrame(seq, index=X.index)], axis=1)
    extra = {}
    if CAMBIOS["H2"]:
        extra.update(razones_y_ventana(X))
    if extra:
        X = pd.concat([X, pd.DataFrame(extra, index=X.index)], axis=1)
    if CAMBIOS["H21"]:
        X = pd.concat([X, pd.DataFrame(tendencia_y_promedio(X), index=X.index)], axis=1)
    return X


def columnas(df: pd.DataFrame) -> list[str]:
    """Todo menos el cliente, el mes (identificaría el mes), la clase y la secuencia de la red (H29)."""
    return [c for c in df.columns if c not in (ID, MES, CLASE) and "__s" not in c]


# ---------------------------------------------------------------- clase, pesos y filas

def multiclase() -> bool:
    return CAMBIOS["H1"] or (CAMBIOS["H20"] and not CAMBIOS["H1b"])


def objetivo(tr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Etiqueta y peso de cada fila de entrenamiento según la clase elegida (B0: BAJA+2 contra el resto)."""
    c = tr[CLASE].astype(str).to_numpy()
    if multiclase():
        y = np.select([c == "BAJA+1", c == "BAJA+2"], [1, 2], 0)
    elif CAMBIOS["H1b"]:
        y = (c != "CONTINUA").astype(int)
    else:
        y = (c == "BAJA+2").astype(int)
    w = np.ones(len(c))
    if CAMBIOS["H20"]:
        w[c == "BAJA+1"] = 0.5
    if CAMBIOS["H14"]:
        w[c == "BAJA+2"] *= 1_072_500 / 27_500
    return y, w


def smote(M: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int):
    """H26: duplica las BAJA+2 con sintéticas, cada una entre una BAJA+2 y una de sus 5 vecinas más cercanas (en
    percentiles por columna; donde falta un valor queda el de la original)."""
    from sklearn.neighbors import NearestNeighbors
    pos = np.flatnonzero(y == y.max())
    P = M[pos]
    R = pd.DataFrame(P).rank(pct=True).fillna(0.5).to_numpy(np.float32)
    vec = NearestNeighbors(n_neighbors=6).fit(R).kneighbors(R, return_distance=False)[:, 1:]
    rng = np.random.default_rng(seed)
    b = P[vec[np.arange(len(P)), rng.integers(0, 5, len(P))]]
    u = rng.random((len(P), 1)).astype(np.float32)
    S = np.where(np.isnan(b), P, P + u * (b - P))
    return np.vstack([M, S]), np.concatenate([y, y[pos]]), np.concatenate([w, w[pos]])


def matriz(tr: pd.DataFrame, cols: list[str], seed: int):
    """Matriz, etiqueta y peso del entrenamiento con los cambios que tocan filas (H5, H19, H26)."""
    y, w = objetivo(tr)
    keep = np.ones(len(tr), dtype=bool)
    if CAMBIOS["H5"]:
        from prepare import muestra_continua
        keep = muestra_continua(tr, 0.2, seed)
    M = tr.loc[keep, cols].to_numpy(np.float32)
    y, w = y[keep], w[keep]
    if CAMBIOS["H19"]:
        viejos = (tr.loc[keep, MES] < max(MESES_ENTRENAMIENTO)).to_numpy()
        for j, c in enumerate(cols):
            if c.startswith("internet"):
                M[viejos, j] = 0 if "_delta" in c else 1  # en 2019 el valor de la mayoría es 1
    if CAMBIOS["H26"]:
        M, y, w = smote(M, y, w, seed)
    return M, y, w


# ---------------------------------------------------------------- modelo general

_DATASETS: dict = {}  # la búsqueda llama muchas veces con los mismos datos: el binning se hace una vez


def _datasets(tr, va, y_va, cols, seed):
    por_semilla = CAMBIOS["H5"] or CAMBIOS["H26"]
    clave = (id(tr), id(va), tuple(cols), seed if por_semilla else None)
    if clave not in _DATASETS:
        _DATASETS.clear()
        M, y, w = matriz(tr, cols, seed)
        dtr = lgb.Dataset(M, y, weight=None if np.all(w == 1) else w, feature_name=cols, free_raw_data=False,
                          params={"feature_pre_filter": False, "verbose": -1})
        y_v = y_va.astype(int) * (2 if multiclase() else 1)  # en multiclase, BAJA+2 es la clase 2
        dva = lgb.Dataset(va[cols].to_numpy(np.float32), y_v, reference=dtr, free_raw_data=False)
        _DATASETS[clave] = (dtr, dva)
    return _DATASETS[clave]


def _auc_baja2(preds, data):
    """Multiclase: early stopping por la AUC de P(BAJA+2) en 201908 (en binario lo hace la métrica auc de LightGBM:
    la etiqueta de validación es siempre BAJA+2)."""
    from prepare import auc
    return "auc_baja2", auc(preds[:, 2], data.get_label() == 2), True


def general(tr, va, y_va, otros, cols, params, seed):
    """Entrena el LightGBM general y devuelve (p_va, [p de cada matriz de otros], rondas). p = P(BAJA+2) en
    multiclase; si no, la probabilidad del positivo."""
    dtr, dva = _datasets(tr, va, y_va, cols, seed)
    p = {**FIJOS, **params, "seed": int(seed)}
    p.update({"objective": "multiclass", "num_class": 3, "metric": "None"} if multiclase()
             else {"objective": "binary", "metric": "auc"})
    m = lgb.train(p, dtr, RONDAS_MAX, valid_sets=[dva], feval=_auc_baja2 if multiclase() else None,
                  callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)])
    pred = lambda X: (lambda q: q[:, 2] if q.ndim == 2 else q)(  # noqa: E731
        m.predict(X[cols].to_numpy(np.float32), num_iteration=m.best_iteration))
    return pred(va), [None if X is None else pred(X) for X in otros], int(m.best_iteration)


def xgboost(tr, va, y_va, te, cols, params, seed):
    """H27: un XGBoost con los mismos datos y parámetros equivalentes, early stopping por AUC en 201908."""
    import xgboost as xgb
    M, y, w = matriz(tr, cols, seed)
    y = (y == y.max()).astype(int)
    dtr = xgb.QuantileDMatrix(M, y, weight=w, feature_names=cols)
    dva = xgb.DMatrix(va[cols].to_numpy(np.float32), y_va.astype(int), feature_names=cols)
    p = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "grow_policy": "lossguide",
         "max_leaves": params["num_leaves"], "max_depth": 0, "eta": params["learning_rate"],
         "subsample": params["bagging_fraction"], "colsample_bytree": params["feature_fraction"],
         "min_child_weight": 1, "seed": int(seed), "verbosity": 0}
    m = xgb.train(p, dtr, RONDAS_MAX, evals=[(dva, "va")], early_stopping_rounds=PACIENCIA, verbose_eval=False)
    r = (0, m.best_iteration + 1)
    pv = m.predict(dva, iteration_range=r)
    pt = None if te is None else m.predict(xgb.DMatrix(te[cols].to_numpy(np.float32), feature_names=cols),
                                           iteration_range=r)
    return pv, pt


def mezcla(a, b):
    """Promedio de los puestos relativos (0 a 1) de los dos modelos."""
    return (rankdata(a) + rankdata(b)) / (2 * len(a))


# ---------------------------------------------------------------- selección de variables (una vez por proceso)

_ELEGIDAS: dict = {}


def seleccion(tr, va, y_va, params) -> list[str]:
    cols = columnas(tr)
    if not (CAMBIOS["H13"] or CAMBIOS["H31"]):
        return cols
    clave = (id(tr), CAMBIOS["H13"], CAMBIOS["H31"])
    if clave not in _ELEGIDAS:
        y, w = objetivo(tr)
        y = (y == y.max()).astype(int)
        M = tr[cols].to_numpy(np.float32)
        q = {**FIJOS, **params, "objective": "binary", "seed": 1, "metric": "None"}
        if CAMBIOS["H13"]:  # canaritos: 50 columnas de ruido; quedan las que juntan más ganancia que el mejor
            ruido = np.random.default_rng(10_050).random((len(M), 50), dtype=np.float32)
            m = lgb.train(q, lgb.Dataset(np.hstack([M, ruido]), y, weight=w), 300)
            g = m.feature_importance("gain")
            elegidas = [c for c, x in zip(cols, g[:len(cols)]) if x > g[len(cols):].max()]
        else:  # SHAP: quedan las que juntan el 99% del SHAP absoluto en una muestra del entrenamiento
            m = lgb.train(q, lgb.Dataset(M, y, weight=w), 300)
            rng = np.random.default_rng(10_031)
            filas = np.union1d(np.flatnonzero(y == 1), rng.choice(len(M), 50_000, replace=False))
            s = np.abs(m.predict(M[filas], pred_contrib=True)[:, :-1]).mean(axis=0)
            o = np.argsort(-s)
            n = int(np.searchsorted(np.cumsum(s[o]) / s.sum(), 0.99)) + 1
            elegidas = [cols[i] for i in sorted(o[:n])]
        _ELEGIDAS[clave] = elegidas if len(elegidas) >= 20 else cols
        print(f"  selección: {len(_ELEGIDAS[clave])} de {len(cols)} variables", flush=True)
    return _ELEGIDAS[clave]


# ---------------------------------------------------------------- especialista (H7, H24)

SEGMENTOS = {
    "internet0": lambda d: d["internet"].to_numpy() == 0,
    "ctrx": lambda d: d["ctrx_quarter"].to_numpy() <= 10,
    "tarjeta": lambda d: (d["Visa_status"].fillna(0).to_numpy() > 0) | (d["Master_status"].fillna(0).to_numpy() > 0),
}
PARAMS_ESP = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 50,
              "feature_fraction": 0.3, "verbose": -1}
RONDAS_ESP, CANARITOS = 200, 50


def segmento_activo() -> str | None:
    return CAMBIOS["H24"] or ("internet0" if CAMBIOS["H7"] else None)


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def repartir(p_gen, p_esp):
    """El nivel lo pone max(general, especialista); el orden, la suma de sus logit (como en la primera)."""
    perfil = np.sort(np.maximum(p_gen, p_esp))[::-1]
    orden = np.argsort(-(logit(p_gen) + logit(p_esp)), kind="stable")
    out = np.empty_like(p_gen)
    out[orden] = perfil
    return out


def especialista(tr, cols, Xs, seed):
    """Binario sobre las filas del segmento de todos los meses de entrenamiento, con variables por canaritos (una vez por
    proceso). Devuelve, para cada X de Xs, (máscara del segmento, probabilidad de esas filas)."""
    seg = SEGMENTOS[segmento_activo()]
    en = seg(tr)
    t = tr[en]
    y, w = objetivo(t)
    y = (y == y.max()).astype(int)
    clave = ("esp", id(tr), segmento_activo())
    if clave not in _ELEGIDAS:
        M = t[cols].to_numpy(np.float32)
        ruido = np.random.default_rng(10_007).random((len(M), CANARITOS), dtype=np.float32)
        m = lgb.train({**PARAMS_ESP, "seed": 9000}, lgb.Dataset(np.hstack([M, ruido]), y, weight=w), RONDAS_ESP)
        g = m.feature_importance("gain")
        el = [cols[i] for i in np.argsort(-g[:len(cols)]) if g[i] > g[len(cols):].max()]
        _ELEGIDAS[clave] = el if len(el) >= 5 else [cols[i] for i in np.argsort(-g[:len(cols)])[:5]]
        print(f"  especialista {segmento_activo()}: {len(t):,} filas, {int(y.sum())} positivos, "
              f"{len(_ELEGIDAS[clave])} variables", flush=True)
    el = _ELEGIDAS[clave]
    m = lgb.train({**PARAMS_ESP, "seed": int(seed)}, lgb.Dataset(t[el].to_numpy(np.float32), y, weight=w), RONDAS_ESP)
    out = []
    for X in Xs:
        if X is None:
            out.append((None, None))
            continue
        k = seg(X)
        out.append((k, m.predict(X.loc[k, el].to_numpy(np.float32))))
    return out


# ---------------------------------------------------------------- segunda etapa en la franja (H8)

FRANJA = (2000, 10000)  # puestos del general dentro de cada mes
PARAMS_2 = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 4, "min_data_in_leaf": 200,
            "feature_fraction": 0.3, "lambda_l2": 20.0, "verbose": -1}
RONDAS_2, VARIABLES_2 = 100, 25
_FUERA: dict = {}


def puesto(p):
    r = np.empty(len(p), int)
    r[np.argsort(-p, kind="stable")] = np.arange(len(p))
    return r


def en_franja(p, excluir=None):
    r = puesto(p)
    k = (r >= FRANJA[0]) & (r < FRANJA[1])
    return k if excluir is None else k & ~excluir


def calibracion(l, y):
    """Regresión logística y ~ c0 + c1 * l, por Newton."""
    A = np.column_stack([np.ones(len(l)), l])
    c = np.zeros(2)
    for _ in range(40):
        mu = 1 / (1 + np.exp(-A @ c))
        c += np.linalg.solve(A.T @ (A * (mu * (1 - mu))[:, None]) + 1e-8 * np.eye(2), A.T @ (y - mu))
    return c


def fuera_de_muestra(tr, va, y_va, cols, params):
    """Cross-fitting por mes: cada mes de entrenamiento puntuado por un general entrenado con los otros (una semilla,
    una vez por proceso). Devuelve {mes: p}."""
    clave = (id(tr), tuple(cols))
    if clave not in _FUERA:
        out = {}
        for m in MESES_ENTRENAMIENTO:
            k = (tr[MES] == m).to_numpy()
            _DATASETS.clear()  # la clave del caché es id(): un subconjunto nuevo puede caer en la dirección del anterior
            _, (p,), _ = general(tr[~k], va, y_va, [tr[k]], cols, params, 800011)
            out[m] = p
        _DATASETS.clear()
        _FUERA[clave] = out
    return _FUERA[clave]


def segunda_etapa(tr, va, y_va, cols, params, seed, pares):
    """Un LightGBM chico aprende, en la franja de los meses de entrenamiento (puntuados fuera de muestra), a corregir al
    general; después reordena la franja de cada mes puntuado. No cambia los valores: cambia a quién le toca cada uno."""
    fuera = fuera_de_muestra(tr, va, y_va, cols, params)
    clase = tr[CLASE].astype(str).to_numpy()
    meses = tr[MES].to_numpy()
    filas, l = [], []
    for m, p in fuera.items():
        idx = np.flatnonzero(meses == m)
        k = en_franja(p) & (clase[idx] != "BAJA+1")
        filas.append(idx[k])
        l.append(logit(p[k]))
    filas, l = np.concatenate(filas), np.concatenate(l)
    y = (clase[filas] == "BAJA+2").astype(float)
    X = tr.iloc[filas]
    clave = ("franja", id(tr))
    if clave not in _ELEGIDAS:  # variables de la franja por canaritos
        M = X[cols].to_numpy(np.float32)
        ruido = np.random.default_rng(10_002).random((len(M), CANARITOS), dtype=np.float32)
        g = lgb.train({**PARAMS_2, "seed": 9000}, lgb.Dataset(np.hstack([M, ruido]), y), RONDAS_2).feature_importance("gain")
        o = [i for i in np.argsort(-g[:len(cols)]) if g[i] > g[len(cols):].max()][:VARIABLES_2]
        _ELEGIDAS[clave] = [cols[i] for i in o] or [cols[i] for i in np.argsort(-g[:len(cols)])[:5]]
        print(f"  franja: {len(y):,} filas, {int(y.sum())} BAJA+2, {len(_ELEGIDAS[clave])} variables", flush=True)
    el = _ELEGIDAS[clave]
    c = calibracion(l, y)
    franjas = [None if D is None else np.flatnonzero(en_franja(p)) for D, p in pares]
    a_puntuar = [(D.iloc[k][el].to_numpy(np.float32), logit(p[k])) for (D, p), k in zip(pares, franjas) if D is not None]
    if CAMBIOS["H30"]:
        puntajes = iter(tabpfn_franja(X[el].to_numpy(np.float32), l, y, a_puntuar, seed))
    else:
        modelo = lgb.train({**PARAMS_2, "seed": int(seed)},
                           lgb.Dataset(X[el].to_numpy(np.float32), y, init_score=c[0] + c[1] * l), RONDAS_2)
        puntajes = iter([modelo.predict(M, raw_score=True) + c[0] + c[1] * lp for M, lp in a_puntuar])
    out = []
    for (D, p), idx in zip(pares, franjas):
        if D is None:
            out.append(None)
            continue
        p = p.copy()
        s = rankdata(next(puntajes)) + rankdata(p[idx])
        nuevo = np.empty(len(idx))
        nuevo[np.argsort(-s, kind="stable")] = np.sort(p[idx])[::-1]
        p[idx] = nuevo
        out.append(p)
    return out


# ---------------------------------------------------------------- red sobre la historia (H29, ChurnNet)

MESES_RED = 6  # la red lee el mes actual y los 5 anteriores de las 152 crudas


def historia_red(df: pd.DataFrame) -> dict:
    """Lags 1 a 5 de las crudas (columnas c__s1 ... c__s5), sólo para la red: LightGBM no las ve."""
    base = base_cols(df)
    ids, per = df[ID].to_numpy(np.int64), _periodo(df[MES].to_numpy(np.int64))
    orden = np.lexsort((per, ids))
    ids_o, v_o = ids[orden], df[base].to_numpy(np.float32)[orden]
    out = {}
    for k in range(1, MESES_RED):
        mismo = np.zeros(len(ids_o), dtype=bool)
        mismo[k:] = ids_o[k:] == ids_o[:-k]
        lag_o = np.full_like(v_o, np.nan)
        lag_o[k:] = v_o[:-k]
        lag_o[~mismo] = np.nan
        lag = np.empty_like(lag_o)
        lag[orden] = lag_o
        out.update({f"{c}__s{k}": lag[:, j] for j, c in enumerate(base)})
    return out


def _tensor(D: pd.DataFrame, crudas: list[str], q: np.ndarray) -> np.ndarray:
    """(clientes, 6 meses, variables), del más viejo al actual; cada variable en su percentil de entrenamiento
    (0 a 1) y -1 donde falta (incluidos los meses antes de que el cliente aparezca)."""
    pasos = [[f"{c}__s{k}" for c in crudas] for k in range(MESES_RED - 1, 0, -1)] + [crudas]
    out = np.empty((len(D), MESES_RED, len(crudas)), dtype=np.float32)
    for t, cols in enumerate(pasos):
        M = D[cols].to_numpy(np.float32)
        for j in range(len(crudas)):
            x = M[:, j]
            r = np.searchsorted(q[:, j], x, side="right") / q.shape[0]
            out[:, t, j] = np.where(np.isnan(x), -1.0, r)
    return out


# La red corre en un proceso aparte que sólo carga torch: en macOS, torch y LightGBM en el mismo proceso chocan (cada uno
# trae su OpenMP) y la GRU se cae con un segfault. Los datos van y vuelven por archivos .npy.
RED_HIJO = r"""
import json, sys
import numpy as np
import torch
d, seed, epocas, paciencia = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
torch.manual_seed(seed)
rng = np.random.default_rng(seed)
disp = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
Xtr = torch.from_numpy(np.load(f"{d}/Xtr.npy"))
ytr = torch.from_numpy(np.load(f"{d}/ytr.npy"))
Xva = torch.from_numpy(np.load(f"{d}/Xva.npy")).to(disp)
y_va = np.load(f"{d}/yva.npy")

def auc(p, y):
    r = np.empty(len(p)); r[np.argsort(p, kind="stable")] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

class Red(torch.nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.gru = torch.nn.GRU(f, h, batch_first=True)
        self.sal = torch.nn.Sequential(torch.nn.Linear(h, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    def forward(self, x):
        _, h = self.gru(x)
        return self.sal(h[-1]).squeeze(-1)

m = Red(Xtr.shape[2]).to(disp)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
perdida = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0, device=disp))

def puntuar(X):
    m.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(m(X[i:i + 16384])) for i in range(0, len(X), 16384)]).cpu().numpy()

mejor, estado, sin_mejora, curva = -1.0, None, 0, []
for _ in range(epocas):
    m.train()
    for b in np.array_split(rng.permutation(len(Xtr)), max(1, len(Xtr) // 4096)):
        b = torch.from_numpy(b)
        opt.zero_grad()
        perdida(m(Xtr[b].to(disp)), ytr[b].to(disp)).backward()
        opt.step()
    a = auc(puntuar(Xva), y_va)
    curva.append(round(a, 4))
    if a > mejor:
        mejor, estado, sin_mejora = a, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0
    else:
        sin_mejora += 1
        if sin_mejora >= paciencia:
            break
m.load_state_dict(estado)
np.save(f"{d}/pva.npy", puntuar(Xva))
try:
    np.save(f"{d}/pte.npy", puntuar(torch.from_numpy(np.load(f"{d}/Xte.npy")).to(disp)))
except FileNotFoundError:
    pass
json.dump({"auc_val_sola": round(mejor, 4), "curva": curva, "dispositivo": str(disp)}, open(f"{d}/info.json", "w"))
"""


def red(tr, va, y_va, te, seed, epocas=20, paciencia=2):
    """H29: una GRU chica sobre los 6 meses de historia, early stopping por AUC en 201908, en un proceso aparte.
    Devuelve (p_va, p_te, info)."""
    import json
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    import config
    crudas = [c for c in base_cols(tr) if f"{c}__s1" in tr.columns]
    q = np.nanquantile(tr[crudas].to_numpy(np.float32), np.linspace(0, 1, 101), axis=0)
    q = np.where(np.isnan(q), np.inf, q)
    y, _ = objetivo(tr)
    (config.CACHE / "v3").mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="red_", dir=config.CACHE / "v3"))
    try:
        np.save(d / "Xtr.npy", _tensor(tr, crudas, q))
        np.save(d / "ytr.npy", (y == y.max()).astype(np.float32))
        np.save(d / "Xva.npy", _tensor(va, crudas, q))
        np.save(d / "yva.npy", np.asarray(y_va, dtype=bool))
        if te is not None:
            np.save(d / "Xte.npy", _tensor(te, crudas, q))
        subprocess.run([sys.executable, "-c", RED_HIJO, str(d), str(int(seed)), str(epocas), str(paciencia)], check=True)
        p_va = np.load(d / "pva.npy").astype(np.float64)
        p_te = np.load(d / "pte.npy").astype(np.float64) if te is not None else None
        return p_va, p_te, json.loads((d / "info.json").read_text())
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- TabPFN en la franja (H30)

# Como la red: TabPFN usa torch, así que corre en un proceso aparte que no carga LightGBM.
TABPFN_HIJO = r"""
import sys
import numpy as np
import torch
from tabpfn import TabPFNClassifier
d, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = TabPFNClassifier(device="mps" if torch.backends.mps.is_available() else "cpu", random_state=seed)
m.fit(np.load(f"{d}/X.npy"), np.load(f"{d}/y.npy"))
for i in range(n):
    np.save(f"{d}/p{i}.npy", m.predict_proba(np.load(f"{d}/Z{i}.npy"))[:, 1])
"""


def tabpfn_franja(M, l, y, a_puntuar, seed, max_filas=10_000):
    """H30: TabPFN aprende la franja (las variables elegidas y el logit de la base) con hasta 10.000 filas (todas las
    BAJA+2 y CONTINUA al azar) y puntúa cada (matriz, logit) de a_puntuar."""
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    import config
    rng = np.random.default_rng(int(seed))
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    filas = np.concatenate([pos, rng.choice(neg, min(len(neg), max_filas - len(pos)), replace=False)])
    (config.CACHE / "v3").mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="tabpfn_", dir=config.CACHE / "v3"))
    try:
        np.save(d / "X.npy", np.column_stack([M[filas], l[filas]]).astype(np.float32))
        np.save(d / "y.npy", y[filas].astype(int))
        for i, (Z, lz) in enumerate(a_puntuar):
            np.save(d / f"Z{i}.npy", np.column_stack([Z, lz]).astype(np.float32))
        subprocess.run([sys.executable, "-c", TABPFN_HIJO, str(d), str(int(seed)), str(len(a_puntuar))], check=True)
        return [np.load(d / f"p{i}.npy") for i in range(len(a_puntuar))]
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- el modelo

def train_and_predict(tr, va, y_va, te, params, seed):
    cols = seleccion(tr, va, y_va, params)
    p_va, (p_te,), rondas = general(tr, va, y_va, [te], cols, params, seed)
    if CAMBIOS["H27"]:
        x_va, x_te = xgboost(tr, va, y_va, te, cols, params, seed)
        p_va, p_te = mezcla(p_va, x_va), (None if te is None else mezcla(p_te, x_te))
    if segmento_activo():
        (kv, ev), (kt, et) = especialista(tr, cols, [va, te], seed)
        p_va[kv] = repartir(p_va[kv], ev)
        if te is not None:
            p_te[kt] = repartir(p_te[kt], et)
    info = {"rondas": rondas, "variables": len(cols)}
    if CAMBIOS["H29"]:
        r_va, r_te, info["red"] = red(tr, va, y_va, te, seed)
        p_va, p_te = mezcla(p_va, r_va), (None if te is None else mezcla(p_te, r_te))
    if CAMBIOS["H8"] or CAMBIOS["H30"]:
        p_va, p_te = segunda_etapa(tr, va, y_va, cols, params, seed, [(va, p_va), (te, p_te)])
    return p_va, p_te, info
