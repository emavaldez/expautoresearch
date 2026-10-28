"""Un issue por hipótesis de hipotesis.tsv (para los comentarios de la página): crea los que faltan y actualiza el
título y el texto de los que ya existen. Anota el número en hipotesis.tsv.

    python tools/issues_hipotesis.py [--solo H2,H3]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys

import comun
import config as C
from ideas import PAGINA

GRUPO = {"base": "Baseline", "principal": "Hipótesis principal", "loop": "Cola del loop",
         "busqueda": "Cómo buscar los hiperparámetros", "loop-metodo": "Sobre el loop", "corte": "Dónde cortar"}
DISENO = {
    "base": "Entrenamiento 201903-201906 completo, búsqueda de 40 trials por AUC en 201908, 10 semillas, test 201910.",
    "principal": "El loop parte del baseline y recorre la cola; el modelo final se compara con el baseline en 201910.",
    "loop": "Un paso del loop: el vigente más este cambio, con los parámetros del vigente y 5 semillas, contra el "
            "vigente en 201908. Si cumple la regla queda; si no, se deshace. 201910 no se mira.",
    "busqueda": "Mismo presupuesto (40 trials) y 10 semillas; contra el baseline en 201910.",
    "loop-metodo": "Se mide sobre el registro del loop o sobre el modelo final.",
    "corte": "Sobre el modelo final; la regla se elige con 201908 y se informa en 201910.",
}


def cuerpo(h: dict) -> str:
    pred = h["prediccion"] + (f" (ΔAUC esperada en 201908: {float(h['pred_auc']):+.4f})" if h.get("pred_auc") else "")
    fuente = h["fuente"] + (f"\n{h['fuente_url']}" if h.get("fuente_url") else "")
    return (f"**Grupo**\n{GRUPO.get(h['fase'], h['fase'])}\n\n**Hipótesis ({h['id']})**\n{h['h1']}\n\n**H0**\nNo hay "
            f"diferencia.\n\n**Predicción**\n{pred}. {h['porque']}\n\n**Diseño**\n{h['cambio']} {DISENO.get(h['fase'], '')}"
            f"\n\n**Fuente**\n{fuente}\n\nProtocolo completo: PROTOCOLO.md. Comentarios sobre la hipótesis o el diseño, "
            f"acá. El resultado se publica acá y en la página ({PAGINA}/#/hipotesis/{h['id']}).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo")
    a = ap.parse_args()
    solo = set(a.solo.split(",")) if a.solo else None
    ruta = comun.RAIZ / "hipotesis.tsv"
    filas = comun.leer_tsv(ruta)
    for h in filas:
        if solo and h["id"] not in solo:
            continue
        titulo = f"[{h['id']}] {h['titulo']}"
        if h.get("issue"):
            subprocess.run(["gh", "issue", "edit", h["issue"], "-R", C.REPO_GITHUB, "--title", titulo,
                            "--body", cuerpo(h)], check=True, capture_output=True)
            print(f"{h['id']}: #{h['issue']} actualizado", flush=True)
        else:
            r = subprocess.run(["gh", "issue", "create", "-R", C.REPO_GITHUB, "--title", titulo, "--body", cuerpo(h),
                                "--label", "idea"], check=True, capture_output=True, text=True)
            h["issue"] = r.stdout.strip().rsplit("/", 1)[-1]
            print(f"{h['id']}: #{h['issue']} creado", flush=True)
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(filas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
