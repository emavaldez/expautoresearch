# Instrucciones para el agente (loop v3)

Mejorás un modelo que marca qué clientes son BAJA+2, con datos de 2019 solamente, siguiendo `PROTOCOLO.md`. Es el
loop de autoresearch de Karpathy: un arnés fijo, un solo archivo editable, una regla de decisión fijada de antemano y
todo registrado.

## Qué es fijo y qué no

- **Fijo (no se toca):** `v3/arnes.py` (datos, meses, semillas, métricas, comparación), `prepare.py`, `config.py`,
  `PROTOCOLO.md` (salvo su sección de cambios, con fecha y motivo).
- **Editable:** sólo `v3/train.py`. Cada idea de la cola es un interruptor de `CAMBIOS` con su código en el mismo archivo.
- **Nunca se mira 201910 para decidir.** Los pasos del loop ni siquiera lo puntúan. 201911 y 201912 no se usan.

## El loop

1. La cola es el grupo `loop` de `hipotesis.tsv`, en el orden de `orden` (por la predicción `pred_auc`, escrita antes de
   correr). Una idea nueva entra a la cola con su predicción escrita ANTES de correrla.
2. `python v3/loop.py` recorre la cola: prende un interruptor sobre el vigente, corre 5 semillas en 201908 y compara con
   el vigente, pareado por semilla. Regla (PROTOCOLO.md): agrega → ΔAUC > 0,001 y > 2 errores estándar; saca algo → no
   baja más de 0,0005; ideas de ganancia (H8, H14, H30) → meseta +2 M y la AUC como freno.
3. Si queda, `v3/train.py` pasa a ser ese archivo y se commitea; si no, se descarta. Todo queda en `v3/loop.tsv`,
   `experimentos_v3/<id>/` y la fila de la hipótesis.
4. Después de cada paso, anotá en `LECCIONES.md` una a tres líneas: qué se probó, qué pasó, por qué creés que pasó.
5. Con la cola vacía: hasta 10 variables propuestas leyendo `LECCIONES.md` (H28), cada una con su predicción y la misma
   regla.
6. Final: `python v3/correr.py FINAL --semillas 10` y `python v3/correr.py H25 --busqueda` (40 trials nuevos), las dos
   comparadas con B0 en 201910. Después, la ablación.

## Reglas

- No se instalan paquetes sin el OK de una persona (H29 y H30 esperan ese OK).
- Sin fuga: para la fila de un cliente en el mes m, `make_features` sólo usa datos hasta m. Nunca la clase. Nada que
  identifique el mes (`foto_mes`) o al cliente entra como variable.
- Los hiperparámetros no se mueven a mano: salen de `v3/correr.py --busqueda`.
- A igual AUC gana lo más simple.
