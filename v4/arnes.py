"""Arnés fijo de v4 (PROTOCOLO.md, sección v4): mejorar el modelo de la primera competencia (E019) con datos de 2019.

- Decide la plata: la meseta del ensamble de semillas, en promedio sobre tres meses con origen móvil (201907, 201908 y
  201909). Cada mes se entrena con lo que se sabía (ventana m - 5 a m - 2, como en el aula), así que hay tres meses
  para decidir en lugar de uno. La AUC y la lista al 3,3% se informan.
- 201910 es el test: sólo se puntúa al final, para comparar. 201911 y 201912 no se usan (su clase depende de 2020).
- Semillas por bloque (prepare.semillas): screening (5) en cada paso, confirmación (10) para lo que pasa la regla,
  búsqueda en v4/hpo.py y final para el modelo final.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))

import config as C  # noqa: E402
import prepare  # noqa: E402

SELECCION = [201907, 201908, 201909]
TEST = [201910]
N_CONFIRMACION = 10
RESULTADOS = RAIZ / "experimentos_v4"
PROBS = C.CACHE / "v4"
METRICAS = ("meseta", "ganancia", "auc")  # ganancia = la lista al 3,3%


def cargar_modulo(ruta: Path, nombre: str = "train_v4"):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def semillas(bloque: str) -> list[int]:
    return prepare.semillas(bloque, N_CONFIRMACION if bloque == "confirmacion" else None)


def evaluar(nombre: str, bloque: str = "screening", meses: list[int] | None = None, log=print) -> dict:
    """Evalúa experimentos_v4/<nombre>/train.py con las semillas del bloque y lo guarda (eval_<bloque>.json); si ya
    está guardado, lo devuelve sin volver a correr."""
    meses = meses or (TEST if bloque == "test" else SELECCION)
    ruta = RESULTADOS / nombre / f"eval_{bloque}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    mod = cargar_modulo(RESULTADOS / nombre / "train.py")
    sem = semillas("screening" if bloque == "test" else bloque)
    res = prepare.evaluar(mod, meses, sem, log=log, guardar=PROBS / nombre / bloque)
    ruta.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return res


def media(res: dict, clave: str = "meseta") -> float:
    return float(np.mean([f["ensamble"][clave] for f in res.values()]))


def comparar(a: dict, b: dict, clave: str = "meseta") -> dict:
    """B contra A en los mismos meses: diferencia del ensamble (media y por mes) y pareada por semilla (cada semilla:
    la media de sus meses)."""
    meses = [m for m in a if m in b]
    por_mes = {m: b[m]["ensamble"][clave] - a[m]["ensamble"][clave] for m in meses}
    comunes = [s for s in a[meses[0]]["semillas"] if s in b[meses[0]]["semillas"]]
    d = np.array([np.mean([b[m]["semillas"][s][clave] - a[m]["semillas"][s][clave] for m in meses]) for s in comunes])
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"delta": float(np.mean(list(por_mes.values()))), "por_mes": por_mes,
            "meses_ganados": int(sum(v > 0 for v in por_mes.values())),
            "pareado": {"media": float(d.mean()), "se": se, "ganadas": int((d > 0).sum()), "n": int(len(d))}}


def todas(a: dict, b: dict) -> dict:
    return {k: comparar(a, b, k) for k in METRICAS}
