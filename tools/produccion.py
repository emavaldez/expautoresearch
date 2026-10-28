"""Producción en 2020 del modelo ya elegido: lo reentrena mes a mes con la ventana de 6 meses, SIN cambiarlo, y mide
qué habría ganado cada lista. Sólo informa: ningún experimento se elige con esto. Lo corre una persona al final.

    python tools/produccion.py E012                    # 202001 a 202009 con la data original
    python tools/produccion.py E012 --modificado       # además 202007-202009 con la data modificada simulada

Escribe experimentos/<id>/produccion.json; la página lo muestra aparte, como "qué habría pasado en 2020".
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import comun
import config as C
import prepare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--semillas", type=int, default=20)
    ap.add_argument("--modificado", action="store_true")
    a = ap.parse_args()
    carpeta = comun.carpeta_experimento(a.id)
    log_f = open(carpeta / "produccion.log", "w", encoding="utf-8")

    def log(msg: str) -> None:
        linea = f"{time.strftime('%H:%M:%S')} {msg}"
        print(linea, flush=True)
        log_f.write(linea + "\n")
        log_f.flush()

    semillas = prepare.semillas("final", a.semillas)
    mod = comun.modulo_experimento(a.id)
    log(f"{a.id}: producción en 2020 con el modelo congelado, {len(semillas)} semillas del bloque final")
    salida = {"id": a.id, "fecha": comun.ahora(), "semillas": semillas,
              "original": prepare.evaluar(mod, C.MESES_PRODUCCION, semillas, df=prepare.cargar(produccion=True), log=log)}
    if a.modificado:
        aula = [202007, 202008, 202009]
        salida["modificado"] = prepare.evaluar(mod, aula, semillas, df=prepare.cargar(modificado=True, produccion=True), log=log)
    (carpeta / "produccion.json").write_text(json.dumps(salida, indent=1) + "\n", encoding="utf-8")
    tot = sum(v["ensamble"]["ganancia"] for v in salida["original"].values())
    log(f"PRODUCCIÓN {a.id}: {comun.millones(tot)} en {len(C.MESES_PRODUCCION)} meses de 2020")
    log_f.close()
    import pagina
    pagina.generar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
