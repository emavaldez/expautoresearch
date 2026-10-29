"""Deja el resultado de cada hipótesis de hipotesis.tsv como comentario en su issue, para que el registro quede ahí
(el texto del issue dice que el resultado se publica en el issue). Un comentario por issue, marcado: si ya está, se
edita en lugar de duplicarlo.

    python tools/comentar_resultados.py [--solo H2,H3] [--ver]     # --ver imprime sin publicar
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import comun
import config as C
from ideas import PAGINA

sys.path.insert(0, str(comun.RAIZ / "v3"))
import loop  # noqa: E402

MARCA = "<!-- resultado-v3 -->"
R = comun.RAIZ / "experimentos_v3"
ARBOL = f"https://github.com/{C.REPO_GITHUB}/tree/main"
BLOB = f"https://github.com/{C.REPO_GITHUB}/blob/main"
ESTADO = {"base": "baseline", "quedó": "queda (entra al modelo)", "descartada": "se descarta",
          "no mejora": "no mejora", "correlación positiva": "correlación positiva",
          "ahorra sin perder": "ahorra sin perder"}


def leer(nombre: str) -> dict:
    return json.loads((R / nombre / "resultado.json").read_text(encoding="utf-8"))


def n(x, d=4, signo=False) -> str:
    return "—" if x is None else (f"{x:+.{d}f}" if signo else f"{x:.{d}f}").replace(".", ",").replace("-", "−")


def m(x, signo=False) -> str:
    return "—" if x is None else f"{n(x / 1e6, 1, signo)} M"


def ic(v: dict, f) -> str:
    return f"[{f(v['ic'][0])}, {f(v['ic'][1])}]" if v.get("ic") else "—"


def por_semilla(v: dict, f) -> str:
    p = v.get("pareado")
    return f"{f(p['media'])} ± {f(p['se']).lstrip('+')} ({p['ganadas']}/{p['n']})" if p else "—"


def regla(idea: str) -> str:
    if idea in loop.SIMPLIFICACIONES:
        return ("saca algo: queda si la AUC no baja más de 0,0005 y la diferencia por semilla no queda por debajo "
                "de −2 EE")
    if idea in loop.POR_GANANCIA:
        return ("pensada para la ganancia: queda si la meseta sube más de 2 M, por semilla más de 2 EE, y la AUC no "
                "baja más de 0,0005")
    return "agrega algo: queda si la AUC sube más de 0,001 y la diferencia por semilla supera 2 EE"


# ---------------------------------------------------------------- tablas por grupo

def tabla_loop(h: dict, pasos: list[dict]) -> str:
    filas = [f for f in pasos if f["id"] == h["id"]]
    vig = leer(filas[0]["vigente"])["val"]
    out = [f"Paso {filas[0]['paso']} del loop, contra el vigente ({filas[0]['vigente']}: AUC {n(vig['auc'])}, meseta "
           f"{m(vig['ganancia_meseta'])} en 201908), con sus parámetros y 5 semillas. Decide 201908; 201910 no se "
           f"mira. Regla: {regla(h['id'])}.", "",
           "| Corrida | AUC 201908 | ΔAUC | ΔAUC por semilla | Δ meseta | Δ meseta por semilla | Decisión |",
           "|---|---|---|---|---|---|---|"]
    for f in filas:
        d = leer(f["corrida"])
        a, g = d["vs_vigente"]["auc"], d["vs_vigente"]["ganancia_meseta"]
        out.append(f"| {f['corrida']} | {n(d['val']['auc'])} | {n(a['delta'], signo=True)} | "
                   f"{por_semilla(a, lambda x: n(x, signo=True))} | {m(g['delta'], True)} | "
                   f"{por_semilla(g, lambda x: m(x, True))} | {f['decision']} |")
    obs = ", ".join(n(float(f["d_auc"]), signo=True) for f in filas)
    out += ["", f"Predicción escrita antes del paso: ΔAUC {n(float(filas[0]['pred_auc']), signo=True)}; dio {obs}."]
    return "\n".join(out)


def tabla_comparacion(nombre: str) -> str:
    d = leer(nombre)
    base = d.get("base")
    filas = [("AUC", "auc", n), ("Meseta", "ganancia_meseta", m), ("Lista (3,3%)", "ganancia_lista", m),
             ("AUC parcial (FPR ≤ 5%)", "auc_parcial", n), ("BAJA+2 en el 1% de arriba", "top1", lambda x: f"{x}")]
    out = ["| | 201908 (validación) | 201910 (test) |", "|---|---|---|"]
    for titulo, k, f in filas:
        out.append(f"| {titulo} | {f(d['val'][k])} | {f(d['test'][k])} |")
    if base:
        for titulo, k, f in filas[:2]:
            fs = lambda x, f=f: f(x, signo=True)  # noqa: E731
            vv, vt = d.get("vs_base_val", {}).get(k, {}), d["vs_base"][k]
            out.append(f"| Δ {titulo if k == 'auc' else titulo.lower()} contra {base} | "
                       f"{fs(vv['delta']) if vv else '—'} | {fs(vt['delta'])} |")
            out.append(f"| IC 95% (remuestreo de clientes) | {ic(vv, fs)} | {ic(vt, fs)} |")
            out.append(f"| Por semilla (media ± EE, ganadas) | {por_semilla(vv, fs)} | {por_semilla(vt, fs)} |")
            if "p" in vt:
                holm = f" (Holm {n(vt['p_holm'], 3)})" if vt.get("p_holm") is not None else ""
                out.append(f"| p | — | {n(vt['p'], 3)}{holm} |")
    texto = "\n".join(f for f in out if not f.endswith("| — | — |"))
    hpo = R / nombre / "hpo.json"
    if hpo.exists():
        p = json.loads(hpo.read_text(encoding="utf-8"))
        q = p["params"]
        objetivo = {"auc": "AUC", "meseta": "meseta"}.get(p["objetivo"], p["objetivo"])
        texto += (f"\n\nBúsqueda: {p['trials']} trials por {objetivo} en 201908; el mejor es el {p['mejor_trial']}: "
                  + ", ".join(f"{k} {n(v, 4) if isinstance(v, float) else v}" for k, v in q.items()) + ".")
    return texto


def tabla_h11() -> str:
    filas = [("B0", "baseline"), ("FINAL0", "B0 + H1b con los parámetros de B0"),
             ("H11", "B0 + H1b con 40 trials nuevos (el final)"), ("ABL-H1b", "el final sin H1b (ablación)")]
    out = ["Corridas en 201910 (10 semillas):", "", "| Corrida | Qué es | AUC | Meseta | Lista |", "|---|---|---|---|---|"]
    for k, que in filas:
        t = leer(k)["test"]
        out.append(f"| {k} | {que} | {n(t['auc'])} | {m(t['ganancia_meseta'])} | {m(t['ganancia_lista'])} |")
    return "\n".join(out) + "\n\n" + tabla_comparacion("H11")


def tabla_h28(filas: list[dict], pasos: list[dict]) -> str:
    hijas = {f["id"]: f for f in filas if f["id"].startswith("H28") and f["id"] != "H28"}
    out = ["| Variable | Issue | ΔAUC en 201908 | Δ meseta | Decisión |", "|---|---|---|---|---|"]
    for p in pasos:
        if p["id"] in hijas:
            out.append(f"| {p['id']} {hijas[p['id']]['titulo']} | #{hijas[p['id']]['issue']} | "
                       f"{n(float(p['d_auc']), signo=True)} | {m(float(p['d_meseta']), True)} | {p['decision']} |")
    return "\n".join(out)


def tabla_corte(nombre: str) -> str:
    d = leer(nombre)
    c = d["corte"]
    lista = {k: d[k]["ganancia_lista"] - d["vs_lista"][k] for k in ("val", "test")}
    return "\n".join([
        "| | 201908 (elige la regla) | 201910 (test) |", "|---|---|---|",
        f"| Envíos con la regla | {c['envios_val']:,} | {c['envios_test']:,} |".replace(",", "."),
        f"| Ganancia con la regla | {m(d['val']['ganancia_lista'])} | {m(d['test']['ganancia_lista'])} |",
        f"| Ganancia de la lista al 3,3% | {m(lista['val'])} | {m(lista['test'])} |",
        f"| Diferencia | {m(d['vs_lista']['val'], True)} | {m(d['vs_lista']['test'], True)} |",
        "", "Regla de decisión: tiene que ganarle a la lista en los dos meses."])


def tabla_h33() -> str:
    d = leer("H33")
    return (f"- Pasos: {d['pasos']} (las variantes de H24 cuentan aparte).\n"
            f"- Spearman entre la ΔAUC predicha y la observada: {n(d['spearman'], 2)} (p {n(d['p'], 4)}).\n"
            f"- Acierto de signo: {n(100 * d['acierto_de_signo'], 0)}%; predije suba en {d['predijo_suba']} y subió en "
            f"{d['subio']}.\n- Media predicha {n(d['media_predicha'], signo=True)}, media observada "
            f"{n(d['media_observada'], signo=True)}.")


def tabla_h34() -> str:
    d = leer("H34")
    perdidos = ", ".join(d["cambios_que_se_habrian_perdido"]) or "ninguno"
    return (f"- Tiempo total del loop: {d['segundos_total'] / 60:.0f} min; ahorrado: {d['segundos_ahorrados'] / 60:.0f} "
            f"min ({n(100 * d['fraccion_ahorrada'], 0)}%).\n- Regla: cortar el paso si la primera semilla viene 0,002 "
            f"o más abajo de la del vigente.\n- Cambios que habrían quedado y se habrían perdido: {perdidos}.")


def detalle(h: dict, filas: list[dict], pasos: list[dict]) -> str:
    i, exp = h["id"], h["experimento"]
    if i == "H11":
        return tabla_h11()
    if i == "H28":
        return tabla_h28(filas, pasos)
    if i == "H33":
        return tabla_h33()
    if i == "H34":
        return tabla_h34()
    if h["fase"] == "corte":
        return tabla_corte(exp)
    if h["fase"] == "loop":
        return tabla_loop(h, pasos)
    return tabla_comparacion(exp)


def cuerpo(h: dict, filas: list[dict], pasos: list[dict]) -> str:
    links = [f"[página]({PAGINA}/#/hipotesis/{h['id']})", f"[protocolo]({BLOB}/PROTOCOLO.md)"]
    if h["fase"] == "loop":
        links.insert(1, f"[registro del loop]({BLOB}/v3/loop.tsv)")
    corridas = list(dict.fromkeys(p["corrida"] for p in pasos if p["id"] == h["id"])) or [h["experimento"]] * bool(
        h["experimento"])
    if h["id"] == "H11":
        corridas = ["B0", "FINAL0", "H11", "ABL-H1b"]
    links[:0] = [f"[experimentos_v3/{c}]({ARBOL}/experimentos_v3/{c})" for c in corridas]
    partes = [MARCA, f"## Resultado: {ESTADO.get(h['estado'], h['estado'])}", h["conclusion"],
              detalle(h, filas, pasos)]
    if h.get("siguiente"):
        partes.append(f"**Qué seguiría:** {h['siguiente']}")
    partes.append("Registro: " + " · ".join(links))
    return "\n\n".join(partes)


def comentario_previo(issue: str) -> int | None:
    r = subprocess.run(["gh", "api", f"repos/{C.REPO_GITHUB}/issues/{issue}/comments", "--paginate"],
                       capture_output=True, text=True, check=True)
    return next((c["id"] for c in json.loads(r.stdout) if MARCA in c["body"]), None)


def publicar(issue: str, texto: str) -> str:
    previo = comentario_previo(issue)
    if previo:
        subprocess.run(["gh", "api", "-X", "PATCH", f"repos/{C.REPO_GITHUB}/issues/comments/{previo}", "-F",
                        "body=@-"], input=texto, text=True, check=True, capture_output=True)
        return "editado"
    subprocess.run(["gh", "issue", "comment", issue, "-R", C.REPO_GITHUB, "--body-file", "-"], input=texto, text=True,
                   check=True, capture_output=True)
    return "comentado"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo")
    ap.add_argument("--ver", action="store_true")
    a = ap.parse_args()
    solo = set(a.solo.split(",")) if a.solo else None
    filas = comun.leer_tsv(comun.RAIZ / "hipotesis.tsv")
    pasos = comun.leer_tsv(loop.REGISTRO)
    for h in filas:
        if solo and h["id"] not in solo:
            continue
        texto = cuerpo(h, filas, pasos)
        if a.ver:
            print(f"===== #{h['issue']}\n{texto}\n")
            continue
        print(f"{h['id']}: #{h['issue']} {publicar(h['issue'], texto)}", flush=True)
        time.sleep(1)  # los límites secundarios de GitHub para crear contenido
    return 0


if __name__ == "__main__":
    sys.exit(main())
