"""Utilidades compartidas por run.py y las herramientas."""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

EXPERIMENTOS = RAIZ / "experimentos"
RESULTADOS = RAIZ / "resultados.tsv"
IDEAS = RAIZ / "ideas.tsv"
ESTADO = RAIZ / "estado.json"

COLUMNAS_RESULTADOS = ["id", "fecha", "idea", "descripcion", "media_seleccion", "delta_vs_vigente", "folds_ganados",
                       "control", "segundos", "estado", "vigente_al_correr"]
COLUMNAS_IDEAS = ["id", "origen", "titulo", "detalle", "fuente", "prioridad", "estado", "experimentos", "actualizado"]


def ahora() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def slug(texto: str, largo: int = 40) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t[:largo].rstrip("-") or "exp"


# ---------------------------------------------------------------- tsv

def leer_tsv(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def escribir_tsv(ruta: Path, filas: list[dict], columnas: list[str]) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for fila in filas:
            w.writerow({k: str(fila.get(k, "")).replace("\t", " ").replace("\n", " ") for k in columnas})


# ---------------------------------------------------------------- estado y experimentos

def estado() -> dict:
    if ESTADO.exists():
        return json.loads(ESTADO.read_text(encoding="utf-8"))
    return {"vigente": None, "lecturas_test": 0}


def guardar_estado(e: dict) -> None:
    ESTADO.write_text(json.dumps(e, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def carpeta_experimento(id_: str) -> Path:
    """'E003' -> experimentos/E003-algo."""
    hits = sorted(EXPERIMENTOS.glob(f"{id_}-*")) + sorted(EXPERIMENTOS.glob(id_))
    if not hits:
        raise FileNotFoundError(f"No existe el experimento {id_}")
    return hits[0]


def resultado(id_: str) -> dict:
    return json.loads((carpeta_experimento(id_) / "resultado.json").read_text(encoding="utf-8"))


def proximo_id() -> str:
    nums = [int(m.group(1)) for p in EXPERIMENTOS.glob("E*") if (m := re.match(r"E(\d+)", p.name))]
    return f"E{(max(nums) + 1) if nums else 0:03d}"


def cargar_modulo(ruta: Path, nombre: str | None = None):
    """Importa un train.py (de la raíz o de la carpeta de un experimento) como módulo independiente."""
    nombre = nombre or f"modelo_{slug(str(ruta), 60).replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def modulo_experimento(id_: str):
    return cargar_modulo(carpeta_experimento(id_) / "train.py", f"modelo_{id_}")


def actualizar_resultado_tsv(id_: str, **cambios) -> None:
    filas = leer_tsv(RESULTADOS)
    for f in filas:
        if f["id"] == id_:
            f.update({k: v for k, v in cambios.items()})
    escribir_tsv(RESULTADOS, filas, COLUMNAS_RESULTADOS)


# ---------------------------------------------------------------- git

def git(*args: str, check: bool = False) -> str:
    r = subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return r.stdout.strip()


def commit(mensaje: str, rutas: list[str | Path]) -> None:
    if not (RAIZ / ".git").exists():
        return
    rutas = [str(p) for p in rutas if Path(p).exists()]
    if not rutas:
        return
    git("add", "--", *rutas)
    if git("diff", "--cached", "--name-only"):
        git("commit", "-q", "-m", mensaje)


def millones(x) -> str:
    try:
        return f"{float(x) / 1e6:,.1f}M".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return "-"
