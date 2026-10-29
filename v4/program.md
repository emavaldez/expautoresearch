# Instrucciones para el agente (loop v4)

Mejorás en plata el modelo de la primera competencia (E019) con datos de 2019, siguiendo la sección v4 de
`PROTOCOLO.md`. Es el mismo loop que v3: arnés fijo, un solo archivo editable, regla fijada de antemano, todo
registrado.

- **Fijo:** `v4/arnes.py`, `prepare.py`, `config.py`, la sección v4 de `PROTOCOLO.md` (salvo su registro de cambios).
- **Editable:** sólo `v4/train.py`. Cada idea es un valor de `CAMBIOS`; con los valores por defecto es E019.
- **201910 no se mira** hasta el final. 201911 y 201912 no se usan.

1. Una idea nueva entra a `v4/cola.tsv` con su predicción (`pred_meseta`, M por mes) escrita antes de correrla.
2. `python v4/loop.py` recorre la cola (las filas sin estado): screening con 5 semillas en 201907-201909 contra el
   vigente; si pasa, confirmación con 10 semillas nuevas; si se confirma, queda y se commitea.
3. `python v4/hpo.py --trials N` busca hiperparámetros sobre el vigente y agrega el ganador a la cola.
4. Después de cada tanda: anotar en `LECCIONES.md` qué se probó, qué pasó y por qué creés que pasó.
5. Sin fuga: `make_features` sólo mira hasta el mes de la fila; nunca la clase; nada que identifique el mes o al
   cliente como variable.
