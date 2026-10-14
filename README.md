# expautoresearch

Método [autoresearch](https://github.com/karpathy/autoresearch) aplicado a churn bancario: un agente de código
(Claude Code) corre experimentos chicos sobre un modelo LightGBM, de a uno, y se queda sólo con lo que mejora.

- **Objetivo:** marcar qué clientes de **202108** son BAJA+2, con 36 meses de historia.
- **Ganancia:** +1.072.500 por cada BAJA+2 estimulado, −27.500 por cada estímulo.
- **Página:** https://emavaldez.github.io/expautoresearch/ — para proponer cosas a probar y ver qué se probó, con qué
  código y qué dio.

## Cómo está armado

| Archivo | Qué es |
|---|---|
| `config.py` | Fijo. Rutas, folds, ganancia, semillas y reglas de decisión. |
| `prepare.py` | Fijo. Carga, clase ternaria (receta del curso), folds con gap de 2 meses, métrica y evaluación. |
| `train.py` | **Lo único que modifica el agente**: `make_features` y `train_and_predict`. |
| `run.py` | Corre un experimento y lo registra todo. |
| `program.md` | Instrucciones del loop para el agente. |
| `experimentos.md` | Plan: qué probar y por qué, con las fuentes. |
| `LECCIONES.md` | Lo aprendido, para no repetir. |
| `ideas.tsv` | Cola de cosas a probar (plan + página + terminal). |
| `resultados.tsv` | Resumen: una fila por experimento. |
| `experimentos/<id>-<slug>/` | Por experimento: `train.py` exacto, `cambios.diff` contra el vigente, `log.txt`, `resultado.json` y, si se confirmó, la comparación. |
| `hpo/<nombre>/` | Búsquedas de hiperparámetros: espacio, trials, estado y log. |
| `sellado/` | Reserva y test de 202108 (sólo resúmenes; las listas con ids quedan en la caché local). |
| `fase0/` | Informe de los datos: clase por mes, variables rotas, deriva. |
| `docs/index.html` | La página (la genera `tools/pagina.py`). |

Herramientas en `tools/`: `comparar.py` (confirmación pareada), `hpo.py` (Optuna), `ideas.py` (cola e issues),
`fuga.py` (control de fuga temporal), `fase0.py`, `sellado.py` (reserva y test), `pagina.py`, `publicar.sh` y
`sinteticos.py` (datos de prueba).

## Validación

- Se entrena con clase hasta m − 2 y se puntúa m, igual que el pedido real (clase hasta 202106, puntuar 202108).
- **Folds de selección:** 202104, 202105 y 202106. Deciden los experimentos.
- **Control:** 202008, el mismo mes calendario que el objetivo un año antes. Se reporta en cada experimento.
- **Reserva:** 202103. Sólo para elegir entre finalistas, una vez.
- **Test sellado:** 202108. El loop nunca ve su clase; se mide al final, con el corte fijado de antemano.
- **Métrica:** ganancia máxima de la curva suavizada (media móvil de 401 envíos) del **ensamble** de semillas.
- **Semillas** en bloques que no se mezclan: screening (5), confirmación (20), búsqueda (20), final.

## Puesta en marcha

1. Python 3.12 o más nuevo y las dependencias: `pip install -r requirements.txt`.
2. Los datos fuera del repositorio. Por defecto:
   - dataset de 36 meses: `~/datos/dataset_36m.csv.gz` (csv, csv.gz o parquet; `numero_de_cliente`, `foto_mes` y
     las variables; si trae `clase_ternaria` se usa, si no se calcula);
   - clase de 202108: `~/datos/clase_202108.csv` (`numero_de_cliente` + clase o 0/1, o sólo la lista de BAJA+2).

   Otras rutas: variables de entorno `EXP_DATOS` y `EXP_TEST`.
3. Poné tu semilla primigenia en `config.SEMILLA_PRIMIGENIA`.
4. Confirmá en `config.py` que los meses de los folds existen en los datos.
5. Abrí Claude Code en esta carpeta y pedile: *"leé program.md y arrancá"*.

La primera corrida arma una caché en `~/.cache/expautoresearch/` (no va al repositorio).

## Proponer cosas para probar

- **Desde la página:** el formulario crea un issue con la etiqueta `idea`.
- **Desde la terminal:** `python tools/ideas.py nueva "título" --detalle "qué y por qué" --prioridad alta`.

El agente las trae con `python tools/ideas.py sync`, las prueba y, si vinieron de un issue, lo comenta con el
resultado y lo cierra.

## Probar todo sin los datos reales

```
python tools/sinteticos.py /tmp/sint
export EXP_DATOS=/tmp/sint/dataset_36m.csv.gz EXP_TEST=/tmp/sint/clase_202108.csv EXP_CACHE=/tmp/sint/cache
python run.py "base" --idea B0 --sin-commit
```

Hacelo en una copia del repositorio: los experimentos sobre datos sintéticos no tienen que quedar en el registro.

## Modelo de referencia

Se agrega cuando esté disponible y se mide con el mismo arnés: ver `base/README.md`.
