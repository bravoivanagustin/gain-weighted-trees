# 0001 — Verificar contra sklearn en capas y no copiar su desempate entre atributos

- **Estado:** Aceptada
- **Spec:** `.claude/specs/01-cart-tree.md` §0, §4.2, §7.2

## Contexto

La primera versión de la spec pedía que el árbol fuera **idéntico** al de `sklearn.tree.DecisionTreeClassifier` "sobre datos sin empates de ganancia", y suponía que con `X ~ normal` no hay empates. Es falso:

- La ganancia de Gini depende solo de los **conteos de clase** a cada lado del corte, no de los valores de `x`. Dos atributos que separan las mismas muestras empatan exactamente, aunque `X` sea continua.
- En los nodos chicos esto es lo normal. En un nodo con 2 muestras de cada clase y `d = 20`, se esperan unos 6–7 atributos que separan perfecto.
- sklearn desempata según su propio orden aleatorio de atributos: un Fisher-Yates parcial sobre un array `features` que persiste entre nodos, con su generador `rand_r`. No lo reproducimos.

Consecuencia: con `d > 1` y árboles profundos, un test campo a campo fallaría casi siempre, y no por bugs.

## Decisión

Igualar a sklearn es un **medio** (detectar bugs), no un fin. Lo que importa es que el árbol sea un CART correcto. La verificación se arma en tres capas:

1. **Igualdad exacta con `d = 1`.** Sin empates entre atributos; los empates dentro de un atributo se resuelven igual que sklearn (umbral menor primero). Se comparan todos los campos del árbol, las importancias y la predicción.
2. **Oráculo nodo por nodo** con `d > 1` y datos reales. Para cada nodo interno se entrena un stump de sklearn (`max_depth=1`) con las muestras del nodo. Su corte tiene que estar en nuestro conjunto de cortes empatados en el máximo (`atol = 1e-12`). Para cada hoja se verifica que se detuvo por la misma regla que sklearn.
3. **Igualdad en distribución** con `max_features < d`. La accuracy media sobre 30 semillas tiene que estar dentro de ±2 desvíos de la de sklearn.

Los empates entre atributos se resuelven con nuestro propio RNG, recorriendo los atributos en el orden de `rng.permutation(d)` y tomando la primera ocurrencia del máximo.

## Alternativas descartadas

- **Portar el RNG de sklearn** (el orden de `rand_r` y del Fisher-Yates persistente). Es frágil, ata el código a detalles internos que cambian entre versiones y no aporta nada a la investigación.
- **Comparar solo la accuracy.** Es demasiado débil: no detecta umbrales mal calculados ni reglas de parada distintas.
- **Generar datos sin empates.** No existe tal cosa con Gini en nodos chicos.

## Consecuencias

- Es una diferencia documentada en el README ("Known differences from scikit-learn").
- El desempate aleatorio uniforme coincide con el límite α → ∞ del selector de la fase 2 (uniforme entre los cortes empatados en el máximo), así que la fase 1 ya se comporta como ese límite.
- Las importancias solo se comparan en la capa 1, porque los empates las cambian legítimamente.
