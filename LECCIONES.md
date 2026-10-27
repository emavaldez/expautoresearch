# Lecciones

Lo que se va aprendiendo, para no repetir. El agente lo lee al empezar cada sesión y agrega una a tres líneas por
experimento: qué se probó, qué pasó y por qué cree que pasó. Formato: `- <id>: <lección>`.

## Datos (fase 0)
- internet: en 2019-2020 es binaria y al revés que en 2021: 1 = usa homebanking (92%, 0,25% de BAJA+2), 0 = grupo chico y riesgoso (8%, 1,3 a 2,4% de BAJA+2).
- Meses rotos (variable casi toda en cero cuando antes no): 201905 y 201910 (rentabilidad, comisiones, márgenes), 201904 (débitos automáticos Visa), 202004 (ccajas_depositos) y 202006 (~30 variables, incluido internet). 202006 es lag de 202007 y último mes con clase para puntuar 202008.
- BAJA+2 por mes: 0,41 a 0,62% en 2019; 202002 0,15% (arranque de la pandemia); 202007 0,35%, 202008 0,28%, 202009 0,34%.
- Montos: la mediana del saldo casi se duplica entre 2019 y 2020 (inflación); transferencias y homebanking suben fuerte con la pandemia.

## Experimentos
- E000 (B0): la base da 370,1 M en las tres listas (183,0 / 79,3 / 107,9), AUC media 0,904. La meseta suma 403,8 M: el corte fijo de 3,3% deja ~34 M, casi todo en agosto (lista de 5.156 contra óptimo de ~2.200). El especialista elige 35-38 variables con canaritos; tarda 14 minutos con 5 semillas.
- E001 (E13b): corte por la proporción que maximizó la ganancia en los dos meses anteriores, puntuados por la base fuera de muestra: 390,0 M (+19,9 M), gana los 3 meses; las listas pasan de ~5.150 a 4.625 / 4.146 / 3.608. El orden es el mismo (misma AUC y meseta): todo el aumento es el corte. Agosto sigue lejos de su meseta (89,5 contra 100,4 M).
