# Instrucciones para el agente

Mejorás un modelo que marca qué clientes son BAJA+2 en el escenario de la Segunda Competencia. El modelo se define SÓLO
con datos de 2019 (el presente es 201912) y se mide con la ganancia de sus listas de 201908, 201909 y 201910, cada una
entrenada con la regla de la ventana de 6 meses (ver config.py). En 2020 el modelo sólo se reentrena.
Trabajás con experimentos chicos, de a uno, y todo queda registrado: el código exacto de cada corrida, su log, su
resultado y la página del proyecto.

## Al empezar cada sesión

1. Leé `README.md`, `config.py` y `prepare.py` (sólo lectura), `train.py`, `experimentos.md` y `LECCIONES.md`.
2. `git pull`, después `python tools/ideas.py sync` (trae las propuestas de la página y las aprobaciones) y
   `python tools/ideas.py lista` (sólo aprobadas y en curso: una propuesta sin la etiqueta `aprobada` NO se corre).
3. Si no existe `fase0/informe.md`: `python tools/fase0.py`, leelo y anotá en `LECCIONES.md` lo que haya que tener en
   cuenta (variables rotas, deriva, estacionalidad). Marcá P1, P2 y P4:
   `python tools/ideas.py estado P1 probada --nota "..."`.
4. Si todavía no hay experimentos, corré la base sin tocar `train.py`:
   `python run.py "base: modelo de la primera competencia" --idea B0`.

## El loop (hasta que una persona lo frene)

1. **Elegí la próxima idea.** Sólo de la cola (`ideas.py lista`): primero las propuestas aprobadas con prioridad alta;
   después el plan por prioridad. Una idea por experimento; si tiene variantes (E1: 3, 6, 12 meses), cada variante es un
   experimento.
2. **Partí del vigente.** Si `train.py` no es el del vigente: `python run.py --restaurar`.
3. **Editá sólo `train.py`** para probar la idea.
4. **Si cambiaste `make_features`:** `python tools/fuga.py`. Si dice FUGA, arreglalo antes de correr.
5. **Corré:** `python run.py "<qué cambia, corto>" --idea <id> --hipotesis "<qué esperás y por qué>"`.
6. **Leé la línea RESUMEN:**
   - `a_confirmar` → `python tools/comparar.py <vigente> <nuevo>`. Si confirma, pasa a ser el vigente solo.
   - `descartado`, `rechazado`, `error` o `fuera_de_presupuesto` → `python run.py --restaurar`.
7. **Anotá en `LECCIONES.md`** una a tres líneas: qué se probó, qué pasó y por qué creés que pasó.
8. **Publicá:** `tools/publicar.sh` (regenera la página, commitea y hace push).

## Reglas

- **Sólo se edita `train.py`.** También podés escribir `LECCIONES.md`, `features_probadas.tsv` y los espacios de
  búsqueda en `hpo/`. `prepare.py` y `config.py` definen el problema y cómo se mide: tocarlos es hacer trampa.
- **Nunca mires 2020.** `prepare.cargar()` trae sólo hasta 201912; no uses `cargar(produccion=True)` ni
  `tools/produccion.py`: eso lo corre una persona al final, sobre el modelo ya elegido, y no decide nada.
- **No instales paquetes.**
- **Sin fuga.** Para la fila de un cliente en el mes m, `make_features` sólo usa datos hasta m. Nunca la clase.
  Nada que identifique el mes (`foto_mes`, mes calendario) entra como variable.
- **Presupuesto.** `run.py` tiene que terminar en `config.PRESUPUESTO_S` segundos. Si no, achicá (submuestreo, menos
  rondas) antes de seguir.
- **Hiperparámetros: no se mueven a mano.** Se diseña un espacio y se busca con `tools/hpo.py` (E8 a E10). El ganador
  se copia a `train.py` y se corre como un experimento más.
- **Simplicidad.** A igual ganancia gana el código más simple. Lo que no mejora se deshace. Si un cambio que SACA
  algo empata en el screening, se confirma con `tools/comparar.py <vigente> <nuevo> --simplificacion`: se adopta si
  no pierde (diferencia pareada mayor que −2 errores estándar).
- **Semillas y corte no se eligen mirando una métrica.** Las semillas salen de los bloques de `config.py`; el corte,
  de la regla `CORTE` de `train.py`.
- **Ideas y comentarios.** Cada idea tiene su issue; `run.py` comenta ahí el resultado. Antes de correr una idea, leé
  los comentarios de su issue (`gh issue view <n> --comments`): pueden traer variantes o advertencias.

## Cómo se decide

- **Screening** (`run.py`, 5 semillas): pasa a confirmación si la ganancia media por mes de test (la de la lista)
  mejora más de 3M contra el vigente y mejora en al menos 2 de los 3 meses.
- **Confirmación** (`tools/comparar.py --modificado`, 20 semillas frescas): la lista del ensamble gana más y la
  diferencia pareada por semilla supera 2 errores estándar. Con la data modificada sólo se informa: si un cambio
  confirma pero pierde mucho con la data modificada, anotalo en `LECCIONES.md`.

## Búsquedas de hiperparámetros

```
python tools/hpo.py correr --base <vigente> --espacio hpo/espacio_base.json --nombre hpo01 --trials 40 --arranque
python tools/hpo.py estado --nombre hpo01          # leé hpo/hpo01/estado.md
python tools/hpo.py proponer --nombre hpo01 hpo/hpo01/propuestas.json   # E10: hasta 2 cada 10 trials
```

Después copiá los mejores parámetros a `PARAMS` de `train.py` y corré `run.py` con `--idea E8` (o la que corresponda).

## Lo que decide una persona

- Cuándo parar.
- Qué propuestas se aprueban (etiqueta `aprobada` en el issue).
- Romper la regla de la ventana de 6 meses (E1b).
- Cuándo se congela el modelo y se corre la producción en 2020 (`tools/produccion.py`).
