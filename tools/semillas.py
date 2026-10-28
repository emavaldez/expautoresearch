"""E14: cuántas semillas hacen falta. Entrena un experimento con N semillas del bloque de búsqueda en los meses de test,
guarda la probabilidad de cada una y mide la lista del ensamble con las primeras 1, 3, 5, 10 y N.

    python tools/semillas.py E019 [--n 20]

Escribe experimentos/<id>/semillas.json. Sirve para decidir cuántas semillas correr en el aula (cada una cuesta tiempo).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import comun
import config as C
import prepare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    mod = comun.modulo_experimento(a.id)
    semillas = prepare.semillas("busqueda", a.n)
    df = prepare.cargar()
    cortes = sorted({1, 3, 5, 10, a.n})
    salida = {"id": a.id, "semillas": semillas, "meses": {}}
    for mes in C.MESES_TEST:
        t0 = time.time()
        train, score, y, info, inter = prepare.datos_fold(df, mod, mes)
        probs = [prepare._llamar(mod, train, score, s, None, inter) for s in semillas]
        fila = {}
        for k in cortes:
            ens = np.mean(probs[:k], axis=0)
            n = prepare.envios(mod, ens, train, score)
            fila[str(k)] = {"lista": prepare.ganancia_en(ens, y, n), "meseta": prepare.ganancia_suavizada(ens, y)[0],
                            "auc": prepare.auc(ens, y)}
        salida["meses"][str(mes)] = fila
        print(f"[{mes}] " + " | ".join(f"{k}: {fila[str(k)]['lista'] / 1e6:,.1f}M" for k in cortes) + f" ({time.time() - t0:.0f} s)", flush=True)
    tot = {k: sum(salida["meses"][str(m)][str(k)]["lista"] for m in C.MESES_TEST) for k in cortes}
    salida["total"] = tot
    print("TOTAL " + " | ".join(f"{k} semillas: {v / 1e6:,.1f}M" for k, v in tot.items()), flush=True)
    (comun.carpeta_experimento(a.id) / "semillas.json").write_text(json.dumps(salida, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
