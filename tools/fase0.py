"""Fase 0: mirar los datos antes de arrancar el loop. Escribe fase0/ (sólo agregados, sin ids).

    python tools/fase0.py

- clase_por_mes.tsv: clientes, CONTINUA / BAJA+1 / BAJA+2 y tasa de BAJA+2 por mes, y la razón de BAJA+2 entre
  cada mes y dos meses antes (sirve para estimar la cantidad de BAJA+2 del mes a puntuar).
- rotas.tsv: variables que en algún mes están todas en cero, todas vacías o constantes, y no lo están en general.
- deriva.tsv: variables cuya mediana o cuartil 3 se aleja más de 25% de la mediana de los 12 meses anteriores.
- informe.md: resumen para leer.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import comun
import prepare
from prepare import CLASE, ID, MES

SALIDA = comun.RAIZ / "fase0"


def main() -> int:
    SALIDA.mkdir(exist_ok=True)
    df = prepare.cargar()
    meses = sorted(df[MES].unique())
    lineas = ["# Fase 0: datos", "", f"{len(df):,} filas, {df.shape[1]} columnas, {len(meses)} meses "
              f"({meses[0]} a {meses[-1]}).", ""]

    # P4: clase por mes
    cnt = df.groupby([MES, CLASE], observed=False).size().unstack(fill_value=0)
    tabla = pd.DataFrame({MES: meses, "clientes": df.groupby(MES).size().reindex(meses).to_numpy()})
    for c in ["CONTINUA", "BAJA+1", "BAJA+2"]:
        tabla[c] = cnt[c].reindex(meses).fillna(0).astype(int).to_numpy() if c in cnt else 0
    tabla["tasa_baja2"] = (tabla["BAJA+2"] / tabla["clientes"]).round(5)
    sin_clase = tabla["CONTINUA"] == 0          # los últimos meses: la clase todavía no se conoce
    tabla["BAJA+2"] = tabla["BAJA+2"].astype(object).where(~sin_clase, "")
    tabla["tasa_baja2"] = tabla["tasa_baja2"].astype(object).where(~sin_clase, "")
    b2 = {m: v for m, v, s in zip(tabla[MES], tabla["BAJA+2"], sin_clase) if not s}
    tabla["razon_vs_2_meses_antes"] = [round(b2[m] / b2[prepare.sumar_meses(m, -2)], 3)
                                       if m in b2 and b2.get(prepare.sumar_meses(m, -2)) else "" for m in tabla[MES]]
    tabla.to_csv(SALIDA / "clase_por_mes.tsv", sep="\t", index=False)
    lineas += ["## Clase por mes", "", "| " + " | ".join(tabla.columns) + " |", "|" + "---|" * tabla.shape[1]]
    lineas += ["| " + " | ".join(str(v) for v in fila) + " |" for fila in tabla.itertuples(index=False)]
    lineas.append("")

    # P1: variables rotas
    variables = [c for c in df.columns if c not in (ID, MES, CLASE) and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(MES)
    ceros = g[variables].agg(lambda s: float((s.fillna(0) == 0).mean()))
    vacias = g[variables].agg(lambda s: float(s.isna().mean()))
    distintos = g[variables].nunique()
    rotas = []
    for c in variables:
        for m in meses:
            motivo = None
            if vacias.at[m, c] == 1 and vacias[c].median() < 1:
                motivo = "todo vacío"
            elif ceros.at[m, c] == 1 and ceros[c].median() < 1:
                motivo = "todo cero"
            elif distintos.at[m, c] <= 1 and distintos[c].median() > 1:
                motivo = "constante"
            if motivo:
                rotas.append({"variable": c, MES: m, "motivo": motivo})
    pd.DataFrame(rotas, columns=["variable", MES, "motivo"]).to_csv(SALIDA / "rotas.tsv", sep="\t", index=False)
    lineas += ["## Variables rotas en algún mes", ""]
    lineas += [f"- {r['variable']} en {r[MES]}: {r['motivo']}" for r in rotas] or ["- ninguna"]
    lineas.append("")

    # P2: deriva
    med = g[variables].median()
    q3 = g[variables].quantile(0.75)
    deriva = []
    for c in variables:
        for i, m in enumerate(meses):
            if i < 12:
                continue
            ref_med = med[c].iloc[i - 12:i].median()
            ref_q3 = q3[c].iloc[i - 12:i].median()
            for nombre, v, ref in (("mediana", med.at[m, c], ref_med), ("q3", q3.at[m, c], ref_q3)):
                if ref and np.isfinite(ref) and abs(ref) > 0 and abs(v / ref - 1) > 0.25:
                    deriva.append({"variable": c, MES: m, "estadistico": nombre, "razon": round(float(v / ref), 3)})
    dv = pd.DataFrame(deriva, columns=["variable", MES, "estadistico", "razon"])
    dv.to_csv(SALIDA / "deriva.tsv", sep="\t", index=False)
    lineas += ["## Deriva (contra la mediana de los 12 meses anteriores)", ""]
    if len(dv):
        por_mes = dv.groupby(MES)["variable"].nunique().sort_index()
        lineas += ["Variables que se corren más de 25% en cada mes:", ""]
        lineas += [f"- {m}: {n}" for m, n in por_mes.items()]
        top = dv.groupby("variable")[MES].nunique().sort_values(ascending=False).head(20)
        lineas += ["", "Las que más meses se corren:", ""] + [f"- {v}: {n} meses" for v, n in top.items()]
    else:
        lineas.append("- ninguna")
    (SALIDA / "informe.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    comun.commit("fase 0: informe de datos", [SALIDA])
    return 0


if __name__ == "__main__":
    sys.exit(main())
