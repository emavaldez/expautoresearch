# expautoresearch

Método [autoresearch](https://github.com/karpathy/autoresearch) aplicado a churn bancario: un agente de código
(Claude Code) corre experimentos chicos sobre un modelo LightGBM, de a uno, y se queda sólo con lo que mejora.

- **Escenario (Segunda Competencia de DMEyF 2026):** el modelo se define sólo con datos de 2019 (el presente es 201912:
  201911 tiene sólo BAJA+1 y 201912 no tiene clase). En 2020 sólo se reentrena todos los meses con los últimos 6 meses;
  en 202007-202009 (el aula) se implementan los cambios que hagan falta con lo que llega.
- **Train y test:** el test son 201908, 201909 y 201910, cada uno entrenado con la regla de la ventana de 6 meses
  (para puntuar m se sabe la clase completa hasta m − 2 y sólo las BAJA+1 de m − 1). Cada experimento se mide con la
  ganancia de sus tres listas. 2020 no decide nada: al final se informa qué habría ganado el modelo elegido.
- **Ganancia:** +1.072.500 por cada BAJA+2 estimulado, −27.500 por cada estímulo.
- **Base:** el modelo entregado en la [primera competencia](https://github.com/emavaldez/primeracompetencia), llevado a
  estos datos (`train.py`).
- **Página:** el libro de experimentos (`docs/`, se publica en Vercel): qué se prueba en cada experimento, el código,
  la fuente, cómo se mide, el resultado, comentarios y propuestas nuevas.

## Cómo está armado

| Archivo | Qué es |
|---|---|
| `config.py` | Fijo. Datos, meses de test, ventana, ganancia, semillas y reglas de decisión. |
| `prepare.py` | Fijo. Datos hasta 201912, clase ternaria, lo que se sabe en cada mes, corte, métrica, evaluación y la data modificada simulada. |
| `train.py` | **Lo único que modifica el agente**: `make_features`, `train_and_predict` y el corte. |
| `run.py` | Corre un experimento y lo registra todo. |
| `program.md` | Instrucciones del loop para el agente. |
| `experimentos.md` | Plan: qué probar, por qué y cómo se mide, con las fuentes. |
| `LECCIONES.md` | Lo aprendido, para no repetir. |
| `ideas.tsv` | La cola: plan y propuestas, con su estado y su issue. |
| `resultados.tsv` | Resumen: una fila por experimento. |
| `anulados/` | E000 y E001, medidos por error en 2020: no cuentan. |
| `experimentos/<id>-<slug>/` | Por experimento: `train.py` exacto, `cambios.diff` contra el vigente, `log.txt`, `resultado.json`. |
| `hpo/<nombre>/` | Búsquedas de hiperparámetros: espacio, trials, estado y log. |
| `docs/` | La página: `index.html`, `estilo.css`, `app.js` y `data/proyecto.json` (lo genera `tools/pagina.py`). |

Herramientas en `tools/`: `comparar.py` (confirmación pareada, también con la data modificada), `hpo.py` (Optuna),
`ideas.py` (cola, propuestas, aprobaciones e issues), `fuga.py` (control de fuga temporal), `fase0.py`, `pagina.py`,
`publicar.sh` y `sinteticos.py` (datos de prueba).

## Proponer experimentos y comentar

- **Proponer:** desde la página, "Proponer un experimento": el formulario abre GitHub con la propuesta escrita (qué,
  por qué, fuente, cómo medirlo) y crea un issue con la etiqueta `propuesta`.
- **Aprobar:** Emmanuel le pone la etiqueta `aprobada`. Recién ahí entra a la cola (`python tools/ideas.py sync`).
- **Comentar:** cada idea tiene su issue; la página muestra los comentarios y el resultado se comenta ahí.

## Puesta en marcha

1. Python 3.12 o más nuevo y las dependencias: `pip install -r requirements.txt`.
2. El dataset de la Segunda Competencia fuera del repositorio, por defecto en
   `~/competencia2/datos/competencia_02_crudo.parquet` (otra ruta: variable `EXP_DATOS`).
3. Abrí Claude Code en esta carpeta y pedile: *"leé program.md y arrancá"*.

La primera corrida arma una caché en `~/.cache/expautoresearch/` (no va al repositorio).
