"""Genera docs/index.html (la página de GitHub Pages) a partir de lo que hay en el repositorio.

    python tools/pagina.py

La corren solos run.py, comparar.py, hpo.py y sellado.py. Los datos van embebidos en el HTML, así que la página
también se puede abrir directo desde el disco.
"""
from __future__ import annotations

import json
import re
import statistics
import sys

import comun
import config as C

PLANTILLA = comun.RAIZ / "tools" / "pagina_plantilla.html"
SALIDA = comun.RAIZ / "docs" / "index.html"


def _cola(texto: str, lineas: int) -> str:
    partes = texto.splitlines()
    if len(partes) <= lineas:
        return texto
    return f"[… {len(partes) - lineas} líneas antes; el log completo está en el repositorio]\n" + "\n".join(partes[-lineas:])


def _experimentos() -> list[dict]:
    salida = []
    for carpeta in sorted(comun.EXPERIMENTOS.glob("E*")):
        ruta = carpeta / "resultado.json"
        if not ruta.exists():
            continue
        r = json.loads(ruta.read_text(encoding="utf-8"))
        folds = []
        for mes, f in (r.get("folds") or {}).items():
            sueltos = [v["ganancia"] for v in f["semillas"].values()]
            folds.append({
                "mes": mes,
                "tipo": "selección" if int(mes) in r.get("folds_seleccion", C.FOLDS_SELECCION) else "control",
                "train_meses": f.get("train_meses"), "baja2": f.get("baja2"),
                "ensamble": f["ensamble"]["ganancia"], "envios": f["ensamble"]["envios"],
                "sueltos_media": statistics.mean(sueltos) if sueltos else None,
                "sueltos_sd": statistics.stdev(sueltos) if len(sueltos) > 1 else 0,
            })
        comp = r.get("comparacion")
        log = (carpeta / "log.txt").read_text(encoding="utf-8") if (carpeta / "log.txt").exists() else ""
        diff = (carpeta / "cambios.diff").read_text(encoding="utf-8") if (carpeta / "cambios.diff").exists() else ""
        if diff.count("\n") > 400:
            diff = "\n".join(diff.splitlines()[:400]) + "\n[… diff recortado; completo en el repositorio]"
        salida.append({
            "id": r["id"], "carpeta": carpeta.name, "fecha": r.get("fecha"), "descripcion": r.get("descripcion"),
            "idea": r.get("idea"), "hipotesis": r.get("hipotesis"), "estado": r.get("estado"),
            "media_seleccion": r.get("media_seleccion"), "control": r.get("control"), "segundos": r.get("segundos"),
            "vigente_al_correr": r.get("vigente_al_correr"), "comparacion": comp,
            "delta": comp["delta_media"] if comp else None,
            "folds_ganados": f"{comp['folds_ganados']}/{len(C.FOLDS_SELECCION)}" if comp else None,
            "confirmacion": r.get("confirmacion"), "error": (r.get("error") or "")[-3000:] or None,
            "folds": folds, "log": _cola(log, 200), "diff": diff,
            "archivos": sorted(p.name for p in carpeta.iterdir() if p.is_file()),
        })
    return salida


def _hpo() -> list[dict]:
    salida = []
    for carpeta in sorted((comun.RAIZ / "hpo").glob("*")):
        trials = comun.leer_tsv(carpeta / "trials.tsv")
        if not carpeta.is_dir() or not trials:
            continue
        mejor = max(trials, key=lambda t: float(t["valor"]))
        arranque = next((t for t in trials if t.get("origen") == "arranque"), None)
        estado_md = (carpeta / "estado.md").read_text(encoding="utf-8") if (carpeta / "estado.md").exists() else ""
        base = re.search(r"Base: (\S+)", estado_md)
        salida.append({"nombre": carpeta.name, "base": base.group(1) if base else None, "trials": len(trials),
                       "mejor": float(mejor["valor"]),
                       "mejores_params": json.dumps(json.loads(mejor["params"]), indent=1, ensure_ascii=False),
                       "arranque": float(arranque["valor"]) if arranque else None})
    return salida


def datos() -> dict:
    return {
        "generado": comun.ahora(),
        "repo": C.REPO_GITHUB,
        "config": {"folds_seleccion": C.FOLDS_SELECCION, "folds_control": C.FOLDS_CONTROL,
                   "folds_reserva": C.FOLDS_RESERVA, "mes_test": C.MES_TEST, "ganancia": C.GAN_ACIERTO,
                   "costo": C.COSTO_ESTIMULO, "max_lecturas": C.MAX_LECTURAS_TEST, "umbral": C.UMBRAL_SCREENING},
        "estado": comun.estado(),
        "ideas": comun.leer_tsv(comun.IDEAS),
        "experimentos": _experimentos(),
        "hpo": _hpo(),
        "sellado": {"reserva": comun.leer_tsv(comun.RAIZ / "sellado" / "reserva.tsv"),
                    "test": comun.leer_tsv(comun.RAIZ / "sellado" / "test.tsv")},
    }


def generar() -> None:
    texto = json.dumps(datos(), ensure_ascii=False).replace("</", "<\\/")
    html = PLANTILLA.read_text(encoding="utf-8").replace("/*__DATOS__*/", texto)
    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(html, encoding="utf-8")
    (SALIDA.parent / ".nojekyll").touch()


if __name__ == "__main__":
    generar()
    print(f"página -> {SALIDA.relative_to(comun.RAIZ)}")
    sys.exit(0)
