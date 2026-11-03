"""Bootstrap de clientes de la diferencia de meseta entre dos ensambles guardados (mismos clientes y mismos meses).

La comparación pareada por semilla del loop mide la suerte de las semillas; ésta mide la de los clientes del mes: se
sortean con reposición los clientes de cada mes y se recalcula la meseta de los dos ensambles.

    MESES=201907,201908,201909 python v4/bootstrap.py V0/screening:R6/screening_p5h3 B2/screening_p5h3:R6/screening_p5h3

Cada argumento es A:B, carpetas de probabilidades en la caché de v4 (arnes.PROBS); informa B - A.
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

def comparar(a, b, B=400, seed=0):
    rng = np.random.default_rng(seed)
    obs, boots = [], np.zeros((B, len(MESES)))
    for j, mes in enumerate(MESES):
        pa, y, ia = cargar(a, mes)
        pb, yb, ib = cargar(b, mes)
        assert np.array_equal(ia, ib) and np.array_equal(y, yb)
        obs.append(prepare.ganancia_suavizada(pb, y)[0] - prepare.ganancia_suavizada(pa, y)[0])
        n = len(y)
        for k in range(B):
            i = rng.integers(0, n, n)
            boots[k, j] = prepare.ganancia_suavizada(pb[i], y[i])[0] - prepare.ganancia_suavizada(pa[i], y[i])[0]
    d = np.mean(obs)
    m = boots.mean(axis=1)
    print(f"{b} - {a}: Δ meseta {d / 1e6:+.2f} M/mes | SE bootstrap de clientes {m.std(ddof=1) / 1e6:.2f} "
          f"| por mes {[round(o / 1e6, 1) for o in obs]} (SE {[round(float(s) / 1e6, 1) for s in boots.std(axis=0, ddof=1)]}) "
          f"| P(Δ>0) {np.mean(m > 0):.2f}")

if __name__ == "__main__":
    for x in sys.argv[1:]:
        comparar(*x.split(":"))
