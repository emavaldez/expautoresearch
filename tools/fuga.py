"""Control de fuga temporal de make_features: las variables de un mes no pueden cambiar si se agregan meses futuros.

    python tools/fuga.py                 # train.py de la raíz, último fold de selección
    python tools/fuga.py experimentos/E004-algo/train.py --mes 202105

Calcula make_features dos veces sobre los mismos clientes del mes m: con datos hasta m y con datos hasta m + 2.
Si alguna columna de las filas de m cambia, make_features está mirando el futuro (o mezcla meses de más).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import comun
import config as C
import prepare
from prepare import CLASE, MES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modelo", nargs="?", default=str(comun.RAIZ / "train.py"))
    ap.add_argument("--mes", type=int, default=C.MESES_TEST[-1])
    a = ap.parse_args()
    mod = comun.cargar_modulo(Path(a.modelo), "modelo_fuga")
    df = prepare.cargar().drop(columns=[CLASE])
    historia = int(getattr(mod, "HISTORIA_MESES", 0))
    desde = max(int(df[MES].min()), prepare.sumar_meses(a.mes, -(historia + 1)))
    corto = df[(df[MES] >= desde) & (df[MES] <= a.mes)]
    largo = df[(df[MES] >= desde) & (df[MES] <= prepare.sumar_meses(a.mes, 2))]
    fc = mod.make_features(corto.copy())
    fl = mod.make_features(largo.copy())
    fc, fl = fc[fc[MES] == a.mes], fl[fl[MES] == a.mes].loc[fc[fc[MES] == a.mes].index]
    distintas = []
    for c in fc.columns:
        x, y = fc[c].to_numpy(), fl[c].to_numpy()
        if np.issubdtype(x.dtype, np.number):
            igual = np.isclose(x.astype(float), y.astype(float), equal_nan=True, rtol=1e-6)
        else:
            igual = (x == y) | (fc[c].isna().to_numpy() & fl[c].isna().to_numpy())
        if not igual.all():
            distintas.append((c, int((~igual).sum())))
    if set(fc.columns) != set(fl.columns):
        print("AVISO: las columnas cambian según los meses que recibe make_features")
    if distintas:
        print(f"FUGA: {len(distintas)} columnas del mes {a.mes} cambian al agregar meses futuros:")
        for c, n in distintas[:30]:
            print(f"  {c}: {n:,} filas")
        return 1
    print(f"sin fuga: las {fc.shape[1]} columnas del mes {a.mes} no cambian al agregar dos meses futuros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
