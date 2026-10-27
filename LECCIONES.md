# Lecciones

Lo que se va aprendiendo, para no repetir. El agente lo lee al empezar cada sesión y agrega una a tres líneas por
experimento: qué se probó, qué pasó y por qué cree que pasó. Formato: `- <id>: <lección>`. Sólo lo que se ve con datos
hasta 201912: nada de 2020 entra acá.

## Datos (fase 0, hasta 201912)
- internet: en 2019 es binaria y al revés que en 2021: 1 = usa homebanking (92%, 0,3% de BAJA+2), 0 = grupo chico y riesgoso (8%, 2,1 a 2,4% de BAJA+2).
- Meses rotos (variable casi toda en cero cuando antes no): 201905 y 201910 (rentabilidad, comisiones, márgenes; en 201910 también chomebanking_transacciones), 201904 (débitos automáticos Visa). Transferencias recibidas y app mobile no existen en la primera mitad de 2019.
- BAJA+2 por mes en 2019: entre 0,41% (201908) y 0,62% (201903); 201910: 0,47%.

## Experimentos
- E000 (B0): la base da 734,0 M en las tres listas de 2019 (199,4 / 250,7 / 283,9), AUC media 0,912; la meseta suma 747,9 M, así que el corte fijo de 3,3% deja sólo ~14 M. El especialista elige ~35-40 variables con canaritos. 11 minutos con 5 semillas.
- E001 (E17): percentil con cero fijo para todos los montos: 731,8 M (−2,2 M), gana 1 de 3 meses, pareado −3,7 ± 0,5 M por mes. Predicción (mejora) errada: en 2019 la deriva entre meses de la ventana es chica y el percentil le saca a los montos su nivel, que sí informa. Queda la defensa para cuando llegue data con escala cambiada, no como regla del modelo.
- E002 (B4): sumar al general el mes m − 1 con BAJA+1 positivas y el resto como CONTINUA: 639,4 M (−94,6 M), pierde los 3 meses, AUC 0,904. Las BAJA+2 de m − 1 (todavía desconocidas) quedan como CONTINUA y son justo el perfil que hay que encontrar: el modelo aprende que ese perfil se queda. Nunca usar el mes parcial con el resto como negativo. Variante pendiente: sólo sus BAJA+1 como positivos extra, sin negativos de ese mes.
- E003 (E3, h = 2): peso 0,5^(edad/2) a los meses del general: 719,7 M (−14,3 M), pierde los 3 meses; la AUC no cambia (0,912) y la meseta baja un poco. En 2019 los tres meses del general se parecen entre sí: darle menos peso a los viejos sólo quita ejemplos.
- E004 (B5): sin segunda etapa: 739,5 M (+5,5 M), gana 2 de 3, pareado +0,4 ± 0,8 M: empata con la base y es más simple y 15% más rápido. Candidato a simplificación: confirmar con 20 semillas que no empeora (la segunda etapa era para la franja dudosa de 2021).
- E005 (B5): sin especialista: 731,8 M (−2,2 M), pareado −1,0 ± 1,3 M, AUC 0,914 (mejor que la base): otro empate. Ni la segunda etapa ni el especialista suman en 2019; falta probar las dos afuera (sólo el general).
