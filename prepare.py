"""Arnés fijo: datos, clase ternaria, train y test, semillas, corte, métrica y evaluación.

El loop de autoresearch NO edita este archivo: define el problema y cómo se mide. Lo único editable es train.py,
que tiene que exponer esta interfaz:

    MESES_ENTRENAMIENTO : int   meses con clase completa que usa, los más recientes dentro de la ventana
                                (para puntuar m: entre m - 5 y m - 2, o sea 4 como máximo; con NACIMIENTO = True,
                                todos los que tengan historia, como el modelo de 201912 que nace con todo 2019)
    HISTORIA_MESES      : int   meses de datos que make_features necesita antes de cada mes de entrenamiento
    make_features(df) -> df     recibe todas las filas de los meses necesarios (<= mes a puntuar), SIN la clase;
                                devuelve una fila por fila de entrada, con el mismo índice
    train_and_predict(train, score, seed, params=None[, intermedio=None]) -> np.ndarray
                                train: features + clase_ternaria; score: features del mes a puntuar; intermedio
                                (opcional): features de m - 1 con la columna baja1 (sus BAJA+1 ya se conocen).
                                Devuelve una probabilidad por fila de score.

Opcional en train.py: CORTE (dict) o corte(prob, train, score) -> int, cuántos estimular (ver envios()).
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


def modificar(df: pd.DataFrame, meses=None) -> pd.DataFrame:
    """Data "modificada" como la del aula, simulada (fija, con semilla) sobre los tres meses de test: lo que nos pueden
    hacer en 202007-202009, ensayado sobre los de 2019 para preparar las herramientas sin mirar 2020.

    Montos con inflación de golpe (x1,30 / x1,35 / x1,40) y 2% de ruido; primer mes: mpayroll en cero y Visa_status
    vacía en el 30%; segundo: ctrx_quarter vacía y cliente_vip ausente; tercero: mcaja_ahorro x100 y cliente_antiguedad
    en años.
    """
    df = df.copy()
    rng = np.random.default_rng(11)
    meses = list(meses or C.MESES_TEST)
    montos = [c for c in df.columns if c.startswith(("m", "Visa_m", "Master_m"))]
    for mes, f in zip(meses, (1.30, 1.35, 1.40)):
        sel = (df[MES] == mes).to_numpy()
        df.loc[sel, montos] = (df.loc[sel, montos].to_numpy() * f * rng.normal(1, 0.02, (sel.sum(), len(montos)))).astype(np.float32)
    m7, m8, m9 = ((df[MES] == m).to_numpy() for m in meses)
    df.loc[m7, "mpayroll"] = 0.0
    df.loc[m7 & (rng.random(len(df)) < 0.3), "Visa_status"] = np.nan
    df.loc[m8, ["ctrx_quarter", "cliente_vip"]] = np.nan
    df.loc[m9, "mcaja_ahorro"] = df.loc[m9, "mcaja_ahorro"] * 100
    df["cliente_antiguedad"] = df["cliente_antiguedad"].astype(np.float32)
    df.loc[m9, "cliente_antiguedad"] = (df.loc[m9, "cliente_antiguedad"] / 12).round(1)
    return df


def cargar(modificado: bool = False, produccion: bool = False) -> pd.DataFrame:
    """Dataset con la clase calculada por presencia (receta del curso), desde la caché.

    Por defecto, sólo hasta config.PRESENTE (201912): la clase queda como se la sabía entonces (201911 sólo BAJA+1,
    201912 sin clase) y nada de 2020 entra. produccion=True lee hasta config.ULTIMO_MES_PRODUCCION, sólo para
    tools/produccion.py. modificado=True altera los meses de test como en el aula (ver modificar)."""
    hasta = C.ULTIMO_MES_PRODUCCION if produccion else C.PRESENTE
    nombre = f"dataset_{hasta}" + ("_modificado" if modificado else "")
    cache = C.CACHE / f"{nombre}.parquet"
    if cache.exists() and (not C.DATOS.exists() or cache.stat().st_mtime >= C.DATOS.stat().st_mtime):
        return pd.read_parquet(cache)
    if modificado:
        df = modificar(cargar(False, produccion), C.MESES_TEST if not produccion else [202007, 202008, 202009])
    else:
        if not C.DATOS.exists():
            raise FileNotFoundError(f"No encuentro el dataset en {C.DATOS} (config.DATOS o la variable EXP_DATOS)")
        t0 = time.time()
        df = _leer(C.DATOS)
        df = df[df[MES] <= hasta].reset_index(drop=True)
        # El id va a entero ANTES de bajar los decimales a float32: viene como double y float32 no representa
        # enteros de más de 2^24 (16,7 millones; hay hasta 78 millones), así que clientes distintos quedaban con el
        # mismo número (214 ids en 2019, con la clase y los lags mezclados).
        df[ID] = df[ID].astype(np.int64)
        df[MES] = df[MES].astype(np.int64)
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        if CLASE in df.columns:
            df = df.drop(columns=[CLASE])
        df[CLASE] = clase_ternaria(df)
        print(f"caché armada: {len(df):,} filas, {df.shape[1]} columnas, {df[MES].nunique()} meses ({time.time() - t0:.0f} s)")
    C.CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
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


# ---------------------------------------------------------------- corte y AUC

def auc(prob: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return float("nan")
    r = pd.Series(np.asarray(prob)).rank().to_numpy()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def envios(mod, prob: np.ndarray, train: pd.DataFrame, score: pd.DataFrame) -> int:
    """Cuántos clientes se estimulan en el mes puntuado, sin mirar su clase.

    train.py puede definir corte(prob, train, score) -> int, o CORTE:
      {"tipo": "fraccion", "f": 0.033}   una proporción fija de los clientes del mes (por defecto)
      {"tipo": "fijo", "envios": n}
      {"tipo": "umbral", "p": 0.025}      los de probabilidad >= p (si train_and_predict devuelve probabilidades calibradas)
    """
    f = getattr(mod, "corte", None)
    if callable(f):
        return max(1, min(int(f(prob, train, score)), len(prob)))
    c = getattr(mod, "CORTE", {"tipo": "fraccion", "f": 0.033})
    if c["tipo"] == "fraccion":
        n = round(c["f"] * len(prob))
    elif c["tipo"] == "fijo":
        n = c["envios"]
    elif c["tipo"] == "umbral":
        n = int((np.asarray(prob) >= c["p"]).sum())
    else:
        raise ValueError(f"CORTE desconocido: {c}")
    return max(1, min(int(n), len(prob)))


# ---------------------------------------------------------------- train y test

def datos_fold(df: pd.DataFrame, mod, mes: int):
    """Arma lo que se sabe al puntuar `mes`: train (meses con clase completa dentro de la ventana), score (el mes) e
    intermedio (m - 1, con la columna baja1: las BAJA+1 ya se conocen, el resto no se sabe)."""
    ultima = sumar_meses(mes, -C.GAP)
    # Modo nacimiento (train.py con NACIMIENTO = True): como el modelo de 201912, que nace con todo lo que hay en 2019;
    # usa todos los meses con clase y con historia, sin la ventana de 6 meses de los reentrenamientos.
    desde_ventana = int(df[MES].min()) if getattr(mod, "NACIMIENTO", False) else sumar_meses(mes, -(C.VENTANA - 1))
    historia = int(getattr(mod, "HISTORIA_MESES", 0))
    n = int(getattr(mod, "MESES_ENTRENAMIENTO", C.VENTANA - C.GAP))
    primero = int(df[MES].min())
    candidatos = [m for m in meses_con_clase(df)
                  if desde_ventana <= m <= ultima and sumar_meses(m, -historia) >= primero]
    if not candidatos:
        raise ValueError(f"No hay meses con clase entre {desde_ventana} y {ultima} con {historia} meses de historia")
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
    previo = sumar_meses(mes, -1)
    intermedio = X[(X[MES] == previo).to_numpy()].copy()
    intermedio["baja1"] = (clase.loc[intermedio.index] == "BAJA+1").to_numpy()
    y = (clase.loc[score.index] == "BAJA+2").to_numpy()
    info = {"train_meses": train_meses, "filas_train": int(len(train)), "clientes": int(len(score)),
            "baja2": int(y.sum()), "columnas": int(score.shape[1])}
    if len(train_meses) < n:
        info["aviso"] = f"pidió {n} meses de entrenamiento y hay {len(train_meses)} con historia completa en la ventana"
    return train, score, y, info, intermedio


def _llamar(mod, train, score, s, params, intermedio):
    import inspect
    kw = {}
    pars = inspect.signature(mod.train_and_predict).parameters
    if params is not None:
        kw["params"] = params
    if "intermedio" in pars:
        kw["intermedio"] = intermedio
    return np.asarray(mod.train_and_predict(train, score, s, **kw), dtype=np.float64)


def evaluar(mod, meses: list[int], semillas_: list[int], df: pd.DataFrame | None = None, log=print,
            guardar: Path | None = None, params: dict | None = None) -> dict:
    """Para cada mes de test: entrena con lo que se sabía, puntúa con cada semilla y arma la lista con el ensamble
    (promedio de probabilidades), que es lo que se entrega. La ganancia que decide es la de esa lista, con la clase
    verdadera del mes. También se guardan la meseta de la curva (lo mejor que se podía con ese orden) y la AUC."""
    df = cargar() if df is None else df
    salida = {}
    for mes in meses:
        t0 = time.time()
        train, score, y, info, intermedio = datos_fold(df, mod, mes)
        log(f"[{mes}] entrena con {info['train_meses']} ({info['filas_train']:,} filas) | puntúa {info['clientes']:,} "
            f"clientes, {info['baja2']} BAJA+2 | {info['columnas']} columnas | features {time.time() - t0:.0f} s")
        if "aviso" in info:
            log(f"[{mes}] AVISO: {info['aviso']}")
        probs, por_semilla = [], {}
        for s in semillas_:
            t1 = time.time()
            p = _llamar(mod, train, score, s, params, intermedio)
            if p.shape != (len(score),):
                raise ValueError(f"train_and_predict devolvió {p.shape}, se esperaban {len(score)} probabilidades")
            k = envios(mod, p, train, score)
            gm, nm = ganancia_suavizada(p, y)
            por_semilla[str(s)] = {"ganancia": ganancia_en(p, y, k), "envios": k, "meseta": gm, "envios_meseta": nm,
                                   "auc": auc(p, y)}
            probs.append(p)
            log(f"[{mes}] semilla {s}: lista {ganancia_en(p, y, k) / 1e6:,.1f}M con {k:,} | meseta {gm / 1e6:,.1f}M en "
                f"{nm:,} | AUC {auc(p, y):.4f} ({time.time() - t1:.0f} s)")
        ens = np.mean(probs, axis=0)
        k = envios(mod, ens, train, score)
        g = ganancia_en(ens, y, k)
        gm, nm = ganancia_suavizada(ens, y)
        aciertos = int(np.asarray(y)[np.argsort(-ens, kind="stable")[:k]].sum())
        log(f"[{mes}] ENSAMBLE de {len(semillas_)}: lista {g / 1e6:,.1f}M ({aciertos} de {int(y.sum())} BAJA+2 con {k:,} "
            f"envíos) | meseta {gm / 1e6:,.1f}M en {nm:,} | AUC {auc(ens, y):.4f}")
        salida[str(mes)] = {**info, "semillas": por_semilla,
                            "ensamble": {"ganancia": g, "envios": k, "aciertos": aciertos, "meseta": gm,
                                         "envios_meseta": nm, "auc": auc(ens, y)},
                            "segundos": round(time.time() - t0, 1)}
        if guardar is not None:
            guardar.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(guardar / f"{mes}.npz", prob=ens, y=y, ids=score[ID].to_numpy())
    return salida


def media_ensamble(res: dict, meses: list[int], clave: str = "ganancia") -> float:
    return float(np.mean([res[str(m)]["ensamble"][clave] for m in meses]))
