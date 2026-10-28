"""Genera los datos de la página (docs/data/proyecto.json). La página (docs/index.html, estilo.css, app.js) es fija y
lee ese archivo; Vercel publica la carpeta docs/.

    python tools/pagina.py

La corren run.py y comparar.py al terminar. Los comentarios se leen en vivo desde los issues de GitHub; acá sólo se
guarda cuántos tiene cada uno.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime

import comun
import config as C

SALIDA = comun.RAIZ / "docs" / "data" / "proyecto.json"
BLOB = f"https://github.com/{C.REPO_GITHUB}/blob/main"

PAPERS = [
    ("A Rolling-Window Framework for Churn Prediction", "https://arxiv.org/abs/2606.06776", "alta", "sirve",
     "Cuántos meses de entrenamiento y ventanas móviles: es la forma del escenario.", ["E1", "E1b", "E2", "E3"]),
    ("How Early Is Early Enough?", "https://arxiv.org/abs/2607.00473", "alta", "sirve",
     "Cuánta historia mirar en las variables, reportando la curva entera.", ["E4", "E5"]),
    ("Seasonal false alarms in churn early-warning systems", "https://arxiv.org/abs/2608.18174", "alta", "a medias",
     "Deltas interanuales y cortes con estacionalidad. Con un solo año (2019) no se puede validar lo interanual, y "
     "nuestra clase no es por umbral de caída; queda la idea de corregir el corte por la tasa reciente.", ["E6", "E7", "P4"]),
    ("Agentic Search Spaces for Tabular ML", "https://arxiv.org/abs/2609.16309", "alta", "sirve",
     "El agente diseña el espacio de hiperparámetros y una búsqueda clásica elige.", ["E8"]),
    ("The Hidden Costs of 99% Accuracy", "https://arxiv.org/abs/2610.00118", "alta", "sirve como método",
     "Control de fuga y descarte de variables redundantes. No usamos SMOTE.", ["P3", "E21"]),
    ("Calibrated, cost-sensitive ensemble for churn in retail banking", "https://doi.org/10.3389/frai.2026.1889639",
     "alta", "sirve", "Corte por probabilidad calibrada, pesos por costo y ensamble de modelos distintos.",
     ["E13", "E14", "E19", "E22", "B3"]),
    ("When Is an LLM Worth It for HPO?", "https://arxiv.org/abs/2606.21641", "media", "sirve como control",
     "El arranque fuerte de la búsqueda es la configuración por defecto, no el modelo de lenguaje.", ["E9"]),
    ("Can LLMs Beat Classical HPO? A Study on autoresearch", "https://arxiv.org/abs/2603.24647", "media", "sirve",
     "Híbrido agente + búsqueda clásica (Centaur).", ["E10"]),
    ("MALMAS: multiagente con memoria para generar variables", "https://arxiv.org/abs/2604.20261", "media",
     "sirve, con expectativa baja", "Variables generadas por roles con memoria y filtro de canaritos.", ["E11", "E12"]),
    ("Data-Driven Telecom Marketing Optimization", "https://arxiv.org/abs/2607.10260", "media", "a medias",
     "Cortes distintos por segmento. El dataset es chico y de telecomunicaciones.", ["E18"]),
    ("ChurnNet", "https://arxiv.org/abs/2606.00169", "media", "no se prueba",
     "Muestra que los modelos de series de tiempo no les ganan a los árboles: respalda no probar redes.", []),
    ("Can We Predict Before Executing ML Agents? (FOREAGENT)", "https://arxiv.org/abs/2601.05930", "media",
     "sirve para el loop", "Ordenar la cola de ideas por la ganancia que se espera.", ["E15"]),
    ("Recovering Wasted Compute in Autoresearch Agents", "https://arxiv.org/abs/2608.10424", "baja",
     "sirve para el loop", "Reglas para no repetir errores y explorar otras ramas.", ["E16"]),
    ("TabFM-Auto", "https://arxiv.org/abs/2609.37989", "baja", "no se prueba",
     "Los modelos fundacionales tabulares no escalan a 4,7 millones de filas y no están instalados.", []),
    ("Explainable AI for churn: CRM integration", "https://arxiv.org/abs/2608.26151", "baja", "no se prueba",
     "Explica predicciones con SHAP; no cambia la ganancia.", []),
    ("Simulated Customers Never Walk Away", "https://arxiv.org/abs/2606.20708", "baja", "no se prueba",
     "Trata de simular usuarios con modelos de lenguaje: otro problema.", []),
    ("Profit Driven Decision Trees for Churn Prediction (ProfTree)", "https://arxiv.org/abs/1712.08101", "agregado",
     "sirve", "Meter la ganancia en el entrenamiento: pesos por costo.", ["E19"]),
    ("Predicting Customer Churn: XGBoost with Temporal Data", "https://arxiv.org/abs/1802.03396", "agregado", "sirve",
     "Variables por ventanas de tiempo y lags por mes calendario.", ["B2", "E4"]),
    ("e-Profits: ganancia por cliente", "https://arxiv.org/abs/2507.08860", "agregado", "no se prueba",
     "Propone ganancia por cliente; acá todos los aciertos valen lo mismo.", []),
]


AULA = {
    "modelo": "E019", "alternativa": "E024", "umbral_saltos": 20, "rango_salto": [0.8, 1.25],
    "saltos_2019": {"normal": [4, 10], "modificado_primer_mes": 37},
    "texto": ("En el aula llegan 202007, 202008 y 202009 de a uno, con la data modificada. El modelo es el vigente, "
              "definido con 2019 y reentrenado cada mes con su ventana. Al lado se calcula siempre la alternativa robusta "
              "(detecta variables rotas y pasa todos los montos a percentil por mes). Si el mes que llega tiene 20 o más "
              "montos cuya mediana se movió fuera de 0,8 a 1,25 contra el mes anterior, se entrega la alternativa en ese mes."),
    # Informe (no decide nada): el modelo congelado reentrenado mes a mes en 2020 y el ensayo del aula con cronómetro.
    "preparacion": [["201912", 4605, 392, 601, 304.6], ["202001", 4717, 354, 521, 259.7], ["202002", 4824, 139, 213, 20.2],
                    ["202003", 4894, 263, 398, 154.7], ["202004", 4911, 288, 524, 181.7], ["202005", 4959, 351, 566, 249.7],
                    ["202006", 5039, 324, 610, 217.8]],
    "ensayo": {"modificado": [["202007", "alternativa", 40, 190.7, 184.1], ["202008", "modelo", 3, 81.5, 81.5],
                              ["202009", "modelo", 1, 115.6, 115.6]],
               "original": [["202007", "modelo", 7, 186.3, 186.3], ["202008", "modelo", 4, 84.8, 84.8],
                            ["202009", "modelo", 0, 117.8, 117.8]],
               "minutos_por_mes": 7.8},
}


def _comentarios() -> dict[str, int]:
    if not shutil.which("gh"):
        return {}
    r = subprocess.run(["gh", "issue", "list", "-R", C.REPO_GITHUB, "--state", "all", "--limit", "500",
                        "--json", "number,comments"], capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    return {str(i["number"]): len(i.get("comments") or []) for i in json.loads(r.stdout)}


def _mes(f: dict) -> dict:
    e = f["ensamble"]
    return {"lista": e["ganancia"], "envios": e["envios"], "aciertos": e.get("aciertos"), "baja2": f["baja2"],
            "clientes": f["clientes"], "meseta": e.get("meseta"), "envios_meseta": e.get("envios_meseta"),
            "auc": e.get("auc"), "train_meses": f.get("train_meses"), "filas_train": f.get("filas_train"),
            "segundos": f.get("segundos")}


def _experimentos() -> list[dict]:
    salida = []
    for carpeta in sorted(comun.EXPERIMENTOS.glob("E*")):
        ruta = carpeta / "resultado.json"
        if not ruta.exists():
            continue
        r = json.loads(ruta.read_text(encoding="utf-8"))
        base = f"{BLOB}/experimentos/{carpeta.name}"
        conf = r.get("confirmacion")
        prod = json.loads((carpeta / "produccion.json").read_text(encoding="utf-8")) if (carpeta / "produccion.json").exists() else None
        salida.append({
            "id": r["id"], "fecha": r["fecha"], "descripcion": r["descripcion"], "idea": r.get("idea", ""),
            "hipotesis": r.get("hipotesis", ""), "estado": r.get("estado"), "vigente_al_correr": r.get("vigente_al_correr"),
            "segundos": r.get("segundos"), "error": (r.get("error") or "")[-1500:],
            "total": r.get("total_test"), "media": r.get("media_test"), "meseta": r.get("media_meseta"),
            "auc": r.get("media_auc"), "semillas": r.get("semillas"),
            "meses": {m: _mes(f) for m, f in (r.get("folds") or {}).items()},
            "modificado": ({"media": r["modificado"]["media_test"],
                            "meses": {m: _mes(f) for m, f in r["modificado"]["folds"].items()}} if r.get("modificado") else None),
            "comparacion": r.get("comparacion"),
            "produccion": ({"original": {m: _mes(f) for m, f in prod["original"].items()},
                            "modificado": {m: _mes(f) for m, f in prod.get("modificado", {}).items()}} if prod else None),
            "confirmacion": ({k: conf.get(k) for k in ("a", "b", "fecha", "ensamble_a", "ensamble_b", "pareado",
                                                       "modificado", "decision")} if conf else None),
            "archivos": {"codigo": f"{base}/train.py",
                         "diff": f"{base}/cambios.diff" if (carpeta / "cambios.diff").exists() else None,
                         "log": f"{base}/log.txt"},
        })
    return salida


def _hipotesis(coment: dict) -> list[dict]:
    """Las hipótesis preregistradas (hipotesis.tsv) con el resultado de su experimento, si ya corrió
    (experimentos_v3/<id>/resultado.json)."""
    res = {}
    for r in sorted((comun.RAIZ / "experimentos_v3").glob("*/resultado.json")):
        d = json.loads(r.read_text(encoding="utf-8"))
        d["carpeta"] = r.parent.name
        res[d["carpeta"]] = d
    out = []
    for f in comun.leer_tsv(comun.RAIZ / "hipotesis.tsv"):
        r = res.get(f.get("experimento") or f["id"])  # con variantes (H24), la que eligió el loop
        if r:
            base = f"{BLOB}/experimentos_v3/{r['carpeta']}"
            r["archivos"] = {"codigo": f"{base}/train.py", "log": f"{base}/log.txt", "busqueda": f"{base}/busqueda.tsv"}
        out.append({**f, "resultado": r, "comentarios": coment.get(f.get("issue", ""), 0)})
    return out


def generar() -> None:
    coment = _comentarios()
    ideas = []
    for f in comun.leer_tsv(comun.IDEAS):
        ideas.append({**{k: f.get(k, "") for k in ("id", "origen", "titulo", "detalle", "fuente", "fuente_url",
                                                  "como_medir", "prioridad", "estado", "issue", "actualizado")},
                      "experimentos": [e for e in f.get("experimentos", "").split(",") if e],
                      "comentarios": coment.get(f.get("issue", ""), 0)})
    lecciones = [l[2:].strip() for l in (comun.RAIZ / "LECCIONES.md").read_text(encoding="utf-8").splitlines()
                 if l.startswith("- ")] if (comun.RAIZ / "LECCIONES.md").exists() else []
    datos = {
        "generado": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "repo": C.REPO_GITHUB,
        "escenario": {"meses_test": C.MESES_TEST, "ventana": C.VENTANA, "gap": C.GAP, "ganancia": C.GAN_ACIERTO,
                      "costo": C.COSTO_ESTIMULO, "umbral_screening": C.UMBRAL_SCREENING,
                      "min_meses_ganados": C.MIN_MESES_GANADOS, "semillas_screening": C.BLOQUES["screening"][1]
                      - C.BLOQUES["screening"][0], "semillas_confirmacion": C.BLOQUES["confirmacion"][1]
                      - C.BLOQUES["confirmacion"][0]},
        "vigente": comun.estado().get("vigente"),
        "experimentos": _experimentos(),
        "ideas": ideas,
        "papers": [{"titulo": t, "url": u, "prioridad": p, "veredicto": v, "para": d, "ideas": i}
                   for t, u, p, v, d, i in PAPERS],
        "lecciones": lecciones,
        "aula": AULA,
        "protocolo": (comun.RAIZ / "PROTOCOLO.md").read_text(encoding="utf-8") if (comun.RAIZ / "PROTOCOLO.md").exists() else "",
        "hipotesis": _hipotesis(coment),
        "loop": comun.leer_tsv(comun.RAIZ / "v3" / "loop.tsv"),
    }
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"página: {SALIDA.relative_to(comun.RAIZ)} ({len(datos['experimentos'])} experimentos, {len(ideas)} ideas)")


if __name__ == "__main__":
    generar()
