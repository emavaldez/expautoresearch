"""Dataset sintético con la misma forma que el real, para probar el pipeline antes de tener los datos.

    python tools/sinteticos.py /tmp/prueba --clientes 15000
    EXP_DATOS=/tmp/prueba/dataset_36m.csv.gz EXP_TEST=/tmp/prueba/clase_202108.csv EXP_CACHE=/tmp/prueba/cache \\
        python run.py "base" --sin-commit

Genera 36 meses (201809 a 202108) y, aparte, las etiquetas de 202108. Los clientes que se van empeoran unos meses
antes; hay aguinaldo en junio y diciembre, inflación en los montos y una variable rota en 202105.
No sirve para sacar conclusiones de modelado: sólo para probar que todo corre.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import comun  # noqa: F401  (pone la raíz en el path)
import prepare


def generar(n0: int, semilla: int = 7):
    rng = np.random.default_rng(semilla)
    meses = []
    m = 201809
    while m <= 202110:
        meses.append(m)
        m = prepare.sumar_meses(m, 1)
    n_max = int(n0 * 2.2)
    edad = rng.integers(20, 80, n_max)
    ingreso = rng.lognormal(11, 0.6, n_max)
    tiene_sueldo = rng.random(n_max) < 0.55
    prestamos = rng.poisson(0.4, n_max)
    vip = rng.random(n_max) < 0.03
    alta = np.full(n_max, -1)
    alta[:n0] = 0
    nuevos = n0
    activo = np.zeros(n_max, bool)
    activo[:n0] = True
    cuenta = np.full(n_max, -1)                 # meses que le quedan a quien decidió irse
    compromiso = rng.normal(0, 1, n_max)
    filas = []
    for t, mes in enumerate(meses):
        if t > 0:
            k = rng.poisson(n0 * 0.025)
            idx = np.arange(nuevos, min(nuevos + k, n_max))
            activo[idx], alta[idx] = True, t
            nuevos += len(idx)
        estacional = 1.3 if mes % 100 in (8, 9) else 1.0
        riesgo = 1 / (1 + np.exp(4.6 - 0.9 * (-compromiso) - 0.4 * (prestamos > 2) + 0.6 * tiene_sueldo))
        decide = activo & (cuenta < 0) & (rng.random(n_max) < riesgo * 0.35 * estacional)
        cuenta[decide] = 3
        compromiso += rng.normal(0, 0.15, n_max) - 0.35 * (cuenta > 0)
        ids = np.flatnonzero(activo)
        yendo = cuenta[ids] > 0
        infl = 1.03 ** t
        aguinaldo = 1.5 if mes % 100 in (6, 12) else 1.0
        sueldo = np.where(tiene_sueldo[ids] & ~(yendo & (cuenta[ids] <= 2)), ingreso[ids] * infl * aguinaldo, 0)
        trx = np.maximum(0, rng.poisson(np.exp(2.5 + 0.5 * compromiso[ids])) - 6 * yendo)
        saldo = np.maximum(0, ingreso[ids] * infl * np.exp(0.4 * compromiso[ids] + rng.normal(0, 0.5, len(ids))) - 4e4 * yendo)
        consumo = np.maximum(0, rng.lognormal(10, 0.8, len(ids)) * infl * (1 - 0.5 * yendo))
        limite = ingreso[ids] * 2 * infl
        filas.append(pd.DataFrame({
            "numero_de_cliente": 10_000_000 + ids, "foto_mes": mes,
            "cliente_edad": edad[ids] + (t // 12), "cliente_antiguedad": t - alta[ids] + 12,
            "cliente_vip": vip[ids].astype(int), "internet": (rng.random(len(ids)) < 0.3).astype(int),
            "active_quarter": (trx > 0).astype(int),
            "mrentabilidad": rng.normal(2000, 1500, len(ids)) * infl + 300 * compromiso[ids],
            "mcuentas_saldo": saldo, "mcaja_ahorro": saldo * rng.uniform(0.3, 0.9, len(ids)),
            "mpayroll": sueldo, "cpayroll_trx": (sueldo > 0).astype(int),
            "ctrx_quarter": trx, "mtarjeta_visa_consumo": consumo,
            "Visa_msaldototal": consumo * rng.uniform(0.5, 1.5, len(ids)), "Visa_mlimitecompra": limite,
            "cprestamos_personales": prestamos[ids], "mprestamos_personales": prestamos[ids] * 1e5 * infl,
            "mcomisiones": rng.gamma(2, 500, len(ids)) * infl,
            "mtransferencias_recibidas": np.maximum(0, rng.normal(3e4, 2e4, len(ids))) * infl * (1 - 0.6 * yendo),
            "ccajas_depositos": 0 if mes == 202105 else rng.poisson(1.5, len(ids)),
        }))
        cuenta[cuenta > 0] -= 1
        se_van = activo & (cuenta == 0)
        activo[se_van] = False
        cuenta[se_van] = -2
    df = pd.concat(filas, ignore_index=True)
    df["clase_ternaria"] = prepare.clase_ternaria(df)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta")
    ap.add_argument("--clientes", type=int, default=15000)
    a = ap.parse_args()
    out = Path(a.carpeta)
    out.mkdir(parents=True, exist_ok=True)
    df = generar(a.clientes)
    test = df[df.foto_mes == 202108][["numero_de_cliente", "clase_ternaria"]]
    test.to_csv(out / "clase_202108.csv", index=False)
    datos = df[df.foto_mes <= 202108].drop(columns=["clase_ternaria"])
    datos.to_csv(out / "dataset_36m.csv.gz", index=False)
    print(f"{len(datos):,} filas, {datos.foto_mes.nunique()} meses -> {out / 'dataset_36m.csv.gz'}")
    print(f"202108: {len(test):,} clientes, {(test.clase_ternaria == 'BAJA+2').sum()} BAJA+2 -> {out / 'clase_202108.csv'}")
    print(df.groupby("foto_mes").clase_ternaria.value_counts().unstack().tail(8))
    return 0


if __name__ == "__main__":
    sys.exit(main())
