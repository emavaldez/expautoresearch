"""Configuración fija del experimento: datos, meses, ganancia, semillas y reglas.

El loop de autoresearch NO edita este archivo. Lo que se puede cambiar por variable de entorno es sólo para
correr las pruebas con datos sintéticos (`tools/sinteticos.py`).
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# ---------------------------------------------------------------- datos
# Dataset de 36 meses (csv, csv.gz o parquet) con una fila por cliente y mes: numero_de_cliente, foto_mes y variables.
DATOS = Path(os.environ.get("EXP_DATOS", "~/datos/dataset_36m.csv.gz")).expanduser()
# Etiquetas del mes de test (202108): numero_de_cliente + clase (o la lista de BAJA+2). Sólo lo lee tools/sellado.py.
ETIQUETAS_TEST = Path(os.environ.get("EXP_TEST", "~/datos/clase_202108.csv")).expanduser()
# Caché local (parquet, probabilidades, listas con ids). Nunca va al repositorio.
CACHE = Path(os.environ.get("EXP_CACHE", "~/.cache/expautoresearch")).expanduser()

# ---------------------------------------------------------------- problema
GAN_ACIERTO = 1_072_500      # por cada BAJA+2 estimulado
COSTO_ESTIMULO = 27_500      # por cada estímulo
VENTANA_SUAVIZADO = 401      # envíos: la ganancia se suaviza para no premiar un pico con suerte
GAP = 2                      # se entrena con clase hasta m - GAP y se puntúa m (igual que el pedido real)

# ---------------------------------------------------------------- folds (fijos antes de arrancar el loop)
FOLDS_SELECCION = [202104, 202105, 202106]   # deciden los experimentos
FOLDS_CONTROL = [202008]                      # mismo mes calendario que el objetivo, un año antes: se reporta
FOLDS_RESERVA = [202103]                      # sólo para elegir entre finalistas (tools/sellado.py reserva)
ENTREGA = {"ultima_clase": 202106, "puntuar": 202108}
MES_TEST = 202108

# ---------------------------------------------------------------- semillas
# Semilla primigenia propia: las semillas son los primos que le siguen. Reemplazala por la tuya antes de empezar.
SEMILLA_PRIMIGENIA = int(os.environ.get("EXP_SEMILLA", "290497"))
# Bloques que no se mezclan: las de screening "ven" muchas decisiones, las de confirmación no.
BLOQUES = {
    "screening": (0, 5),       # cada corrida de run.py
    "confirmacion": (5, 25),   # tools/comparar.py
    "busqueda": (25, 45),      # tools/hpo.py
    "final": (45, 145),        # reserva y test
}

# ---------------------------------------------------------------- reglas
PRESUPUESTO_S = int(os.environ.get("EXP_PRESUPUESTO", "1200"))  # tope de una corrida de run.py (se corta al 150%)
UMBRAL_SCREENING = 3_000_000   # mejora mínima de la media de los folds de selección para pasar a confirmación
MIN_FOLDS_GANADOS = 2          # y tiene que mejorar en al menos esta cantidad de folds de selección
MAX_LECTURAS_TEST = 3          # lecturas del test sellado para candidatos (la base de referencia no cuenta)

REPO_GITHUB = "emavaldez/expautoresearch"
