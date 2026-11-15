# 0002 — `sample_weight` entero desde la fase 1; bootstrap como pesos

- **Estado:** Aceptada
- **Spec:** `.claude/specs/01-cart-tree.md` §2, §5.2, §5.4, §6

## Contexto

La fase 4 usa `RandomForestClassifier` como control. Antes de afirmar que el método reduce la varianza, nuestro bosque con α → ∞ tiene que reproducir a RF en distribución.

RF **no** repite filas para el bootstrap. Le pasa a cada árbol `sample_weight = np.bincount(indices, minlength=n)` (`sklearn/ensemble/_forest.py`, `_parallel_build_trees`). Esto importa porque `min_samples_leaf` y `min_samples_split` cuentan **filas**, no peso (`_splitter.pyx`): una fila con peso 5 es una sola fila. Con filas repetidas, las copias contarían para esas reglas y los árboles serían distintos de los de RF.

## Decisión

- El árbol acepta `sample_weight` desde la fase 1, **solo con enteros ≥ 0** (float64 con valores enteros). Cualquier otra cosa da `ValueError`.
- Las filas con peso 0 no entran al árbol, igual que en sklearn.
- Las reglas `min_samples_*` cuentan filas. Gini, `value` e importancias usan conteos ponderados.
- Las fracciones de `min_samples_*` se resuelven contra `X.shape[0]`, incluidas las filas con peso 0, igual que sklearn.
- El bosque (fase 3) pasa el bootstrap como pesos `bincount`, nunca como filas repetidas.

## Alternativas descartadas

- **Bootstrap con filas repetidas.** Es más simple, pero rompe el control contra RF (ver el contexto).
- **Pesos reales arbitrarios.** Con pesos no enteros, las sumas en float64 dependen del orden y pueden romper empates que sklearn ve como exactos, o al revés. Con enteros, los conteos ponderados son exactos y los empates se resuelven igual que en sklearn. Nadie en el proyecto necesita pesos no enteros.
- **`class_weight` y `min_weight_fraction_leaf`.** Quedan fuera: no hacen falta para RF ni para los experimentos.

## Consecuencias

- La capa 1 y la capa 2 de la verificación corren cada caso sin pesos y con pesos de bootstrap.
- `Tree` guarda `weighted_n_node_samples`, igual que sklearn.
- Es una diferencia documentada en el README.
