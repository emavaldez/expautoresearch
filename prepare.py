"""Arnés fijo: datos, clase ternaria, folds, semillas, métrica y evaluación.

El loop de autoresearch NO edita este archivo: define el problema y cómo se mide. Lo único editable es train.py,
que tiene que exponer esta interfaz:

    MESES_ENTRENAMIENTO : int   meses con clase que usa (los más recientes que permite el gap)
    HISTORIA_MESES      : int   meses de datos que make_features necesita antes de cada mes de entrenamiento
    make_features(df) -> df     recibe todas las filas de los meses necesarios (<= mes a puntuar), SIN la clase;
                                devuelve una fila por fila de entrada, con el mismo índice
    train_and_predict(train, score, seed, params=None) -> np.ndarray
                                train: features + columna clase_ternaria; score: features del mes a puntuar.
                                Devuelve una probabilidad por fila de score.

Opcional en train.py:
    CORTE = {"tipo": "ratio"} | {"tipo": "fijo", "envios": n} | {"tipo": "umbral", "p": 0.0256}
            cómo se decide cuántos estimular en el mes sin clase (ver tools/sellado.py). Por defecto "ratio".
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

ID, MES, CLASE = "numero_de_cliente", "foto_mes", "clase_ternaria"


# ---------------------------------------------------------------- meses

def sumar_meses(mes: int, k: int) -> int:
    """202101 + (-2) -> 202011."""
    a, b = divmod(int(mes), 100)
    t = a * 12 + (b - 1) + k
    return (t // 12) * 100 + t % 12 + 1


def _periodo(mes):
    return (mes // 100) * 12 + mes % 100


# ---------------------------------------------------------------- clase ternaria

def clase_ternaria(df: pd.DataFrame) -> pd.Series:
    """Receta del curso: un cliente es BAJA+1 si no está el mes siguiente, BAJA+2 si está el siguiente y no el
    subsiguiente, CONTINUA si está en los dos. El último mes queda sin clase; el anteúltimo sólo puede ser BAJA+1."""
    per = _periodo(df[MES].to_numpy().astype(np.int64))
    ids = df[ID].to_numpy()
    orden = np.lexsort((per, ids))
    p, i = per[orden], ids[orden]
    p1 = np.full(len(p), np.nan)
    p2 = np.full(len(p), np.nan)
    mismo1 = i[1:] == i[:-1]
    p1[:-1][mismo1] = p[1:][mismo1]
    mismo2 = i[2:] == i[:-2]
    p2[:-2][mismo2] = p[2:][mismo2]
    ultimo = p.max()
    anteultimo = ultimo - 1
    clase = np.full(len(p), None, dtype=object)
    clase[p < anteultimo] = "CONTINUA"
    clase[(p < ultimo) & (np.isnan(p1) | (p + 1 < p1))] = "BAJA+1"
    clase[(p < anteultimo) & (p + 1 == p1) & (np.isnan(p2) | (p + 2 < p2))] = "BAJA+2"
    salida = np.empty(len(p), dtype=object)
    salida[orden] = clase
    return pd.Series(pd.Categorical(salida, categories=["CONTINUA", "BAJA+1", "BAJA+2"]), index=df.index)


# ---------------------------------------------------------------- datos

def _leer(ruta: Path) -> pd.DataFrame:
    if ruta.suffix == ".parquet":
        return pd.read_parquet(ruta)
    return pd.read_csv(ruta, engine="pyarrow")


def cargar() -> pd.DataFrame:
    """Dataset con la clase calculada, desde la caché (se arma la primera vez). La clase del mes de test y de los
    meses posteriores queda siempre vacía: el modelo no la puede ver."""
    cache = C.CACHE / "dataset.parquet"
    if cache.exists() and (not C.DATOS.exists() or cache.stat().st_mtime >= C.DATOS.stat().st_mtime):
        return pd.read_parquet(cache)
    if not C.DATOS.exists():
        raise FileNotFoundError(f"No encuentro el dataset en {C.DATOS} (config.DATOS o la variable EXP_DATOS)")
    t0 = time.time()
    df = _leer(C.DATOS)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df[ID] = df[ID].astype(np.int64)
    df[MES] = df[MES].astype(np.int64)
    if CLASE not in df.columns:
        df[CLASE] = clase_ternaria(df)
    else:
        df[CLASE] = pd.Categorical(df[CLASE], categories=["CONTINUA", "BAJA+1", "BAJA+2"])
    df.loc[df[MES] >= C.MES_TEST, CLASE] = np.nan
    C.CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    print(f"caché armada: {len(df):,} filas, {df.shape[1]} columnas, {df[MES].nunique()} meses ({time.time() - t0:.0f} s)")
    return df


def meses_con_clase(df: pd.DataFrame) -> list[int]:
    return sorted(int(m) for m in df.loc[df[CLASE].notna(), MES].unique())


# ---------------------------------------------------------------- semillas

def _es_primo(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    k = 3
    while k * k <= n:
        if n % k == 0:
            return False
        k += 2
    return True


def semillas(bloque: str = "screening", n: int | None = None) -> list[int]:
    """Primos desde la semilla primigenia (inclusive si es primo), cortados por bloque."""
    a, b = C.BLOQUES[bloque]
    if n is not None:
        b = min(b, a + n)
    out, k = [], C.SEMILLA_PRIMIGENIA
    while len(out) < b:
        if _es_primo(k):
            out.append(k)
        k += 1
    return out[a:b]


def muestra_continua(train: pd.DataFrame, fraccion: float, seed: int) -> np.ndarray:
    """Submuestreo de CONTINUA por hash de (cliente, mes, semilla): la misma semilla elige las mismas filas en
    cualquier variante, así que las comparaciones quedan pareadas. Devuelve una máscara de filas a usar."""
    if fraccion >= 1:
        return np.ones(len(train), dtype=bool)
    h = pd.util.hash_pandas_object(train[[ID, MES]], index=False, hash_key=f"{int(seed) % 10**16:016d}")
    u = h.to_numpy(dtype=np.uint64) / float(2**64)
    return (train[CLASE].to_numpy() != "CONTINUA") | (u < fraccion)


# ---------------------------------------------------------------- métrica

def curva_ganancia(prob: np.ndarray, es_baja2: np.ndarray) -> np.ndarray:
    """Ganancia acumulada estimulando de a uno, de mayor a menor probabilidad."""
    orden = np.argsort(-np.asarray(prob), kind="stable")
    g = np.where(np.asarray(es_baja2)[orden], C.GAN_ACIERTO, -C.COSTO_ESTIMULO)
    return np.cumsum(g, dtype=np.float64)


def ganancia_suavizada(prob: np.ndarray, es_baja2: np.ndarray, ventana: int = C.VENTANA_SUAVIZADO) -> tuple[float, int]:
    """Máximo de la curva de ganancia suavizada con una media móvil de `ventana` envíos, y en cuántos envíos está."""
    cum = curva_ganancia(prob, es_baja2)
    if len(cum) <= ventana:
        i = int(np.argmax(cum))
        return float(cum[i]), i + 1
    suave = np.convolve(cum, np.ones(ventana) / ventana, mode="valid")
    i = int(np.argmax(suave))
    return float(suave[i]), i + ventana // 2 + 1


def ganancia_en(prob: np.ndarray, es_baja2: np.ndarray, envios: int) -> float:
    cum = curva_ganancia(prob, es_baja2)
    envios = max(1, min(int(envios), len(cum)))
    return float(cum[envios - 1])


# ---------------------------------------------------------------- folds

def datos_fold(df: pd.DataFrame, mod, mes: int, ultima_clase: int | None = None):
    """Arma train y score para puntuar `mes` con clase hasta `ultima_clase` (por defecto mes - GAP)."""
    ultima = sumar_meses(mes, -C.GAP) if ultima_clase is None else ultima_clase
    historia = int(getattr(mod, "HISTORIA_MESES", 0))
    n = int(getattr(mod, "MESES_ENTRENAMIENTO", 1))
    primero = int(df[MES].min())
    # un mes entra al entrenamiento sólo si tiene detrás toda la historia que pide make_features
    candidatos = [m for m in meses_con_clase(df) if m <= ultima and sumar_meses(m, -historia) >= primero]
    if not candidatos:
        raise ValueError(f"No hay meses con clase <= {ultima} con {historia} meses de historia detrás")
    train_meses = candidatos[-n:]
    desde = sumar_meses(train_meses[0], -historia)
    sub = df[(df[MES] >= desde) & (df[MES] <= mes)]
    clase = sub[CLASE]
    X = mod.make_features(sub.drop(columns=[CLASE]))
    if len(X) != len(sub) or not X.index.equals(sub.index):
        raise ValueError("make_features tiene que devolver una fila por fila de entrada, con el mismo índice")
    if CLASE in X.columns:
        raise ValueError("make_features no puede devolver la clase")
    en_train = X[MES].isin(train_meses).to_numpy()
    train = X[en_train].copy()
    train[CLASE] = clase[en_train].to_numpy()
    score = X[(X[MES] == mes).to_numpy()]
    y = (clase.loc[score.index] == "BAJA+2").to_numpy()
    info = {"train_meses": train_meses, "filas_train": int(len(train)), "clientes": int(len(score)),
            "baja2": int(y.sum()), "columnas": int(score.shape[1])}
    if len(train_meses) < n:
        info["aviso"] = f"pidió {n} meses de entrenamiento y hay {len(train_meses)} con historia completa"
    return train, score, y, info


def evaluar(mod, meses: list[int], semillas_: list[int], df: pd.DataFrame | None = None, log=print,
            guardar: Path | None = None, params: dict | None = None) -> dict:
    """Entrena y puntúa cada mes con cada semilla. Mide cada modelo suelto y el ensamble (promedio de
    probabilidades), que es lo que se entrega."""
    df = cargar() if df is None else df
    salida = {}
    for mes in meses:
        t0 = time.time()
        train, score, y, info = datos_fold(df, mod, mes)
        log(f"[{mes}] entrena con {info['train_meses']} ({info['filas_train']:,} filas) | puntúa {info['clientes']:,} "
            f"clientes, {info['baja2']} BAJA+2 | {info['columnas']} columnas | features {time.time() - t0:.0f} s")
        if "aviso" in info:
            log(f"[{mes}] AVISO: {info['aviso']}")
        probs, por_semilla = [], {}
        for s in semillas_:
            t1 = time.time()
            p = np.asarray(mod.train_and_predict(train, score, s, params) if params is not None
                           else mod.train_and_predict(train, score, s), dtype=np.float64)
            if p.shape != (len(score),):
                raise ValueError(f"train_and_predict devolvió {p.shape}, se esperaban {len(score)} probabilidades")
            g, n = ganancia_suavizada(p, y)
            por_semilla[str(s)] = {"ganancia": g, "envios": n}
            probs.append(p)
            log(f"[{mes}] semilla {s}: {g / 1e6:,.1f}M en {n:,} envíos ({time.time() - t1:.0f} s)")
        ens = np.mean(probs, axis=0)
        g, n = ganancia_suavizada(ens, y)
        log(f"[{mes}] ENSAMBLE de {len(semillas_)}: {g / 1e6:,.1f}M en {n:,} envíos")
        salida[str(mes)] = {**info, "semillas": por_semilla, "ensamble": {"ganancia": g, "envios": n},
                            "segundos": round(time.time() - t0, 1)}
        if guardar is not None:
            guardar.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(guardar / f"{mes}.npz", prob=ens, y=y, ids=score[ID].to_numpy())
    return salida


def media_ensamble(res: dict, meses: list[int]) -> float:
    return float(np.mean([res[str(m)]["ensamble"]["ganancia"] for m in meses]))


def predecir(mod, ultima_clase: int, mes: int, semillas_: list[int], df: pd.DataFrame | None = None, log=print):
    """Entrena con clase hasta `ultima_clase` y puntúa `mes` (sin clase). Devuelve ids y probabilidad del ensamble."""
    df = cargar() if df is None else df
    train, score, _, info = datos_fold(df, mod, mes, ultima_clase=ultima_clase)
    log(f"[{mes}] entrena con {info['train_meses']} ({info['filas_train']:,} filas), puntúa {info['clientes']:,}")
    probs = [np.asarray(mod.train_and_predict(train, score, s), dtype=np.float64) for s in semillas_]
    return score[ID].to_numpy(), np.mean(probs, axis=0)
