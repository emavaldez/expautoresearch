# Modelo de referencia

Pendiente. Cuando esté disponible se mide en 202108 con el mismo arnés y aparece en la página, en la sección
del test sellado. Su lectura no cuenta contra las lecturas de los candidatos.

- **Si llega como lista de clientes a estimular** (un CSV con `numero_de_cliente`), copiala acá como `lista.csv`:

  ```
  python tools/sellado.py lista base/lista.csv --nombre referencia
  ```

- **Si llega como código Python con la interfaz de `train.py`** (`make_features`, `train_and_predict`,
  `MESES_ENTRENAMIENTO`, `HISTORIA_MESES`), copialo acá como `modelo.py`:

  ```
  python tools/sellado.py test --modulo base/modelo.py --referencia --nombre referencia --envios <N>
  ```

  También se puede correr en los mismos folds que los experimentos para compararlo pareado: copiarlo a
  `train.py` y correrlo con `run.py "modelo de referencia" --idea REF`.

- **Si llega en otro lenguaje** (por ejemplo R): correrlo aparte para generar la lista de 202108 y usar el primer
  caso.
