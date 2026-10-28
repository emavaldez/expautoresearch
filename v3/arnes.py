"""Arnés fijo de los experimentos v3 (ver PROTOCOLO.md): datos, meses, semillas, métricas y comparación.

El loop NO edita este archivo. Lo único editable es v3/train.py, que expone:

    MESES_ENTRENAMIENTO : list[int]  meses con clase completa con los que entrena (dentro de 201901 a 201906)
    PARAMS              : dict       hiperparámetros (salen de hpo.py)
    ESPACIO             : dict       espacio de búsqueda de hpo.py: nombre -> ("log"|"float"|"int"|"intlog", a, b)
    FIJOS               : dict       parámetros que la búsqueda no toca (bagging_freq, objetivo, ...)
    make_features(df) -> df          todas las filas de 201901 a 201910, SIN la clase; una fila por fila de entrada,
                                     mismo índice. Nada que identifique el mes entra como variable.
    train_and_predict(tr, va, y_va, te, params, seed) -> (p_va, p_te, info)
        tr: features + clase_ternaria de MESES_ENTRENAMIENTO; va: features de 201908 e y_va (sus BAJA+2), sólo para
        el early stopping; te: features de 201910 sin clase (None en la búsqueda). info: dict (rondas, variables, ...).

201911 y 201912 no se usan (quedan para el tablero). 201910 no decide nada: sólo se lee para informar.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import config as C  # noqa: E402
import prepare  # noqa: E402

ID, MES, CLASE = prepare.ID, prepare.MES, prepare.CLASE

PRIMER_MES, VALIDACION, TEST = 201901, 201908, 201910
ENTRENAMIENTO_PERMITIDO = [201901, 201902, 201903, 201904, 201905, 201906]
SEMILLAS_FINAL = 10      # baseline y modelos que se comparan en 201910
SEMILLAS_LOOP = 5        # cada paso del loop (las 5 primeras de las 10: quedan pareadas)
FRACCION_LISTA = 0.033   # el corte del aula: 3,3% de los clientes del mes
FPR_PARCIAL = 0.05       # AUC parcial: falsos positivos hasta 5%
BOOTSTRAP = 1000

RESULTADOS = RAIZ / "experimentos_v3"
PROBS = C.CACHE / "v3"


def semillas(n: int = SEMILLAS_FINAL) -> list[int]:
    """Los n primeros primos desde la semilla primigenia: las mismas para todos los experimentos."""
    out, k = [], C.SEMILLA_PRIMIGENIA
    while len(out) < n:
        if prepare._es_primo(k):
            out.append(k)
        k += 1
    return out


# ---------------------------------------------------------------- datos

def datos(mod, log=print):
    """(tr, va, y_va, te, y_te) con las features de train.py. y_te queda en el arnés: train.py nunca lo ve."""
    meses = list(mod.MESES_ENTRENAMIENTO)
    if not set(meses) <= set(ENTRENAMIENTO_PERMITIDO):
        raise ValueError(f"MESES_ENTRENAMIENTO fuera de {ENTRENAMIENTO_PERMITIDO}: {meses}")
    df = prepare.cargar()
    df = df[(df[MES] >= PRIMER_MES) & (df[MES] <= TEST)].reset_index(drop=True)
    clase = df.pop(CLASE)
    t0 = time.time()
    X = mod.make_features(df)
    if not X.index.equals(df.index):
        raise ValueError("make_features tiene que devolver una fila por fila de entrada, con el mismo índice")
    del df
    log(f"features: {X.shape[1]} columnas, {len(X):,} filas ({time.time() - t0:.0f} s)")
    tr = X[X[MES].isin(meses)].copy()
    tr[CLASE] = clase[tr.index].to_numpy()
    va = X[X[MES] == VALIDACION]
    te = X[X[MES] == TEST]
    y_va = (clase[va.index] == "BAJA+2").to_numpy()
    y_te = (clase[te.index] == "BAJA+2").to_numpy()
    return tr, va, y_va, te, y_te


# ---------------------------------------------------------------- métricas

def curva(p, y) -> np.ndarray:
    return prepare.curva_ganancia(p, y)


def meseta(cum: np.ndarray, ventana: int = C.VENTANA_SUAVIZADO) -> tuple[float, int]:
    """Pico de la curva promediada en ventanas de 401 envíos (200 antes y 200 después de cada punto)."""
    c = np.concatenate([[0.0], np.cumsum(cum)])
    suave = (c[ventana:] - c[:-ventana]) / ventana
    i = int(np.argmax(suave))
    return float(suave[i]), i + ventana // 2 + 1


def auc_parcial(p, y, fpr_max: float = FPR_PARCIAL) -> float:
    """Área bajo la ROC hasta fpr_max, dividida por fpr_max (1 = perfecto en la zona de arriba)."""
    y = np.asarray(y, dtype=bool)
    orden = np.argsort(-np.asarray(p), kind="stable")
    ys = y[orden]
    tpr = np.concatenate([[0.0], np.cumsum(ys) / ys.sum()])
    fpr = np.concatenate([[0.0], np.cumsum(~ys) / (~ys).sum()])
    k = int(np.searchsorted(fpr, fpr_max, side="right"))
    x = np.append(fpr[:k], fpr_max)
    t = np.append(tpr[:k], np.interp(fpr_max, fpr, tpr))
    return float(np.trapezoid(t, x) / fpr_max)


def metricas(p, y) -> dict:
    y = np.asarray(y, dtype=bool)
    cum = curva(p, y)
    n_lista = max(1, round(FRACCION_LISTA * len(p)))
    m, n_m = meseta(cum)
    top = np.argsort(-np.asarray(p), kind="stable")[: math.ceil(0.01 * len(p))]
    return {
        "auc": prepare.auc(p, y),
        "ganancia_meseta": m, "envios_meseta": n_m,
        "ganancia_lista": float(cum[n_lista - 1]), "envios_lista": n_lista,
        "ganancia_maxima": float(cum.max()), "envios_maxima": int(np.argmax(cum)) + 1,
        "auc_parcial": auc_parcial(p, y),
        "top1": int(y[top].sum()),
        "baja2": int(y.sum()), "clientes": int(len(y)),
    }


# ---------------------------------------------------------------- corrida

def correr(mod, nombre: str, n_semillas: int, params: dict | None = None, log=print, con_test: bool = True) -> dict:
    """Entrena con cada semilla, guarda las probabilidades (fuera del repo) y devuelve las métricas por semilla y
    del ensamble (promedio de probabilidades) en validación y test. Con con_test=False (los pasos del loop) 201910 ni
    se puntúa: el paso sólo ve validación."""
    params = dict(mod.PARAMS if params is None else params)
    tr, va, y_va, te, y_te = datos(mod, log)
    if not con_test:
        te = None
    ss = semillas(n_semillas)
    pv, pt, infos = [], [], []
    for s in ss:
        t0 = time.time()
        a, b, info = mod.train_and_predict(tr, va, y_va, te, params, s)
        info = dict(info or {}, segundos=round(time.time() - t0, 1))
        pv.append(np.asarray(a, dtype=np.float64))
        infos.append(info)
        mv = metricas(pv[-1], y_va)
        msg = f"semilla {s}: val AUC {mv['auc']:.4f} meseta {mv['ganancia_meseta'] / 1e6:,.1f}M"
        if te is not None:
            pt.append(np.asarray(b, dtype=np.float64))
            mt = metricas(pt[-1], y_te)
            msg += f" | test AUC {mt['auc']:.4f} meseta {mt['ganancia_meseta'] / 1e6:,.1f}M"
        log(f"{msg} | {info}")
    PROBS.mkdir(parents=True, exist_ok=True)
    extra = {"test": np.array(pt), "y_test": y_te} if pt else {}
    np.savez_compressed(PROBS / f"{nombre}.npz", semillas=np.array(ss), val=np.array(pv), y_val=y_va, **extra)
    out = {"semillas": ss, "params": params, "info": infos, "val": metricas(np.mean(pv, axis=0), y_va),
           "por_semilla": {"val": [metricas(x, y_va) for x in pv]}}
    if pt:
        out["test"] = metricas(np.mean(pt, axis=0), y_te)
        out["por_semilla"]["test"] = [metricas(x, y_te) for x in pt]
    return out


def probs(nombre: str) -> dict:
    z = np.load(PROBS / f"{nombre}.npz")
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- comparación

def _pareado(a: list[dict], b: list[dict], k: str) -> dict:
    from scipy.stats import wilcoxon
    d = np.array([x[k] - y[k] for x, y in zip(a, b)])
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    p = float(wilcoxon(d, alternative="greater").pvalue) if len(d) > 1 and np.any(d != 0) else 1.0
    return {"media": float(d.mean()), "se": se, "ganadas": int((d > 0).sum()), "n": int(len(d)), "p_wilcoxon": p}


def _remuestreo(pa, pb, y, reps: int = BOOTSTRAP, semilla: int = C.SEMILLA_PRIMIGENIA) -> dict:
    """Bootstrap por clientes del mes, pareado (los mismos clientes para los dos modelos)."""
    rng = np.random.default_rng(semilla)
    n = len(y)
    da, dm = np.empty(reps), np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, n, n)
        ya = y[i]
        da[r] = prepare.auc(pa[i], ya) - prepare.auc(pb[i], ya)
        dm[r] = meseta(curva(pa[i], ya))[0] - meseta(curva(pb[i], ya))[0]
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
    return {"auc": {"ic": q(da), "p": float((da <= 0).mean())},
            "ganancia_meseta": {"ic": q(dm), "p": float((dm <= 0).mean())}}


def comparar(a: str, b: str, mes: str = "test", n_semillas: int | None = None, bootstrap: bool = True) -> dict:
    """a contra b (nombres de corridas guardadas), en 'val' o 'test': diferencia del ensamble, intervalo por
    remuestreo de clientes y diferencia pareada por semilla (las semillas en común, en orden)."""
    A, B = probs(a), probs(b)
    n = min(len(A["semillas"]), len(B["semillas"])) if n_semillas is None else n_semillas
    if list(A["semillas"][:n]) != list(B["semillas"][:n]):
        raise ValueError("las corridas no comparten semillas: no se pueden parear")
    y = A[f"y_{mes}"]
    pa, pb = A[mes][:n].mean(axis=0), B[mes][:n].mean(axis=0)
    ma, mb = metricas(pa, y), metricas(pb, y)
    sa = [metricas(x, y) for x in A[mes][:n]]
    sb = [metricas(x, y) for x in B[mes][:n]]
    boot = _remuestreo(pa, pb, y) if bootstrap else None
    out = {}
    for k in ("auc", "ganancia_meseta"):
        out[k] = {"delta": ma[k] - mb[k], "pareado": _pareado(sa, sb, k)}
        if boot:
            out[k].update(ic=boot[k]["ic"], p=boot[k]["p"])
    for k in ("ganancia_lista", "auc_parcial", "top1", "ganancia_maxima"):
        out[k] = {"delta": ma[k] - mb[k], "pareado": _pareado(sa, sb, k)}
    return out


def holm(ps: dict[str, float]) -> dict[str, float]:
    """p-valores ajustados por Holm (paso a paso, monótonos)."""
    orden = sorted(ps, key=ps.get)
    m, acum, out = len(orden), 0.0, {}
    for i, k in enumerate(orden):
        acum = max(acum, min(1.0, (m - i) * ps[k]))
        out[k] = acum
    return out
