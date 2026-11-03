"""Bootstrap de clientes de la diferencia de meseta entre dos ensambles guardados (mismos clientes y mismos meses).

La comparación pareada por semilla del loop mide la suerte de las semillas; ésta mide la de los clientes del mes: se
sortean con reposición los clientes de cada mes y se recalcula la meseta de los dos ensambles.

    MESES=201907,201908,201909 python v4/bootstrap.py V0/screening:R6/screening_p5h3 B2/screening_p5h3:R6/screening_p5h3

Cada argumento es A:B, carpetas de probabilidades en la caché de v4 (arnes.PROBS); informa B - A. Con
GUARDAR=experimentos_v4/RUIDO/resultado.json suma cada comparación a ese archivo (lo lee la página).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arnes  # noqa: E402
import prepare  # noqa: E402

P = arnes.PROBS
MESES = tuple(os.environ.get("MESES", ",".join(map(str, arnes.SELECCION))).split(","))

def cargar(ruta, mes):
    z = np.load(P / ruta / f"{mes}.npz")
    o = np.argsort(z["ids"])
    return z["prob"][o], z["y"][o].astype(bool), z["ids"][o]

def comparar(a: str, b: str, meses=MESES, B: int = 400, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obs, boots = [], np.zeros((B, len(meses)))
    for j, mes in enumerate(meses):
        pa, y, ia = cargar(a, mes)
        pb, yb, ib = cargar(b, mes)
        assert np.array_equal(ia, ib) and np.array_equal(y, yb)
        obs.append(prepare.ganancia_suavizada(pb, y)[0] - prepare.ganancia_suavizada(pa, y)[0])
        n = len(y)
        for k in range(B):
            i = rng.integers(0, n, n)
            boots[k, j] = prepare.ganancia_suavizada(pb[i], y[i])[0] - prepare.ganancia_suavizada(pa[i], y[i])[0]
    m = boots.mean(axis=1)
    r = {"a": a.split("/")[0], "b": b.split("/")[0], "bloque_a": a.split("/")[1], "bloque_b": b.split("/")[1],
         "meses": [int(x) for x in meses], "delta": float(np.mean(obs)), "se": float(m.std(ddof=1)),
         "por_mes": [float(o) for o in obs], "se_por_mes": [float(s) for s in boots.std(axis=0, ddof=1)],
         "prob_positiva": float(np.mean(m > 0)), "reps": B}
    print(f"{b} - {a}: Δ meseta {r['delta'] / 1e6:+.2f} M/mes | SE bootstrap de clientes {r['se'] / 1e6:.2f} "
          f"| por mes {[round(o / 1e6, 1) for o in r['por_mes']]} (SE {[round(s / 1e6, 1) for s in r['se_por_mes']]}) "
          f"| P(Δ>0) {r['prob_positiva']:.2f}", flush=True)
    return r


if __name__ == "__main__":
    import json
    salida = os.environ.get("GUARDAR")
    res = [comparar(*x.split(":")) for x in sys.argv[1:]]
    if salida:
        ruta = Path(salida)
        previos = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else []
        clave = lambda r: (r["a"], r["b"], tuple(r["meses"]))
        nuevos = {clave(r) for r in res}
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps([r for r in previos if clave(r) not in nuevos] + res, indent=1) + "\n", encoding="utf-8")
