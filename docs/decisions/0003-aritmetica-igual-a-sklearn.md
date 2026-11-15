# 0003 — Reproducir la aritmética de sklearn

- **Estado:** Aceptada
- **Spec:** `.claude/specs/01-cart-tree.md` §4.1, §4.2, §5.1, §5.2, §5.4

## Contexto

La capa 1 de la verificación (0001) exige igualdad exacta con `d = 1`. Para eso no alcanza con la misma fórmula matemática: hace falta el **mismo redondeo**. Un corte que en sklearn queda justo por encima de un límite y en el nuestro justo por debajo cambia todo el subárbol.

Datos verificados en el código de sklearn 1.8 (`.venv/Lib/site-packages/sklearn/tree/`):

| Qué | Dónde | Valor |
|---|---|---|
| Umbral mínimo entre valores consecutivos | `_partitioner.pxd:15` | `FEATURE_THRESHOLD = 1e-7`, de tipo `float32_t`: la comparación se hace en **float32** |
| Umbral del corte | `_splitter.pyx:459` | punto medio en double entre los dos valores |
| Clave para elegir el corte | `_criterion.pyx:175` | proxy `-wR·G_R - wL·G_L` (se maximiza) |
| Mejora para `min_impurity_decrease` | `_criterion.pyx:208` | `(W/W_root)·(G_t - wR/W·G_R - wL/W·G_L)` |
| Gini | `_criterion.pyx` | `1 - Σc²/(W·W)` |
| `min_samples_split` fraccional | `_classes.py:333-336` | `max(2, ceil(frac·n))`, después `max(mss, 2·msl)` |
| Reglas de parada | `_tree.pyx:232-254` | profundidad, `min_samples_split`, `2·min_samples_leaf`, nodo puro (`<= EPSILON`) |
| Importancias | `_tree.pyx:1265` | con `weighted_n_node_samples` |
| `tree_.value` | — | guarda fracciones, no conteos |

## Decisión

- Las comparaciones contra `FEATURE_THRESHOLD` se hacen en float32 explícito: `np.float32(1e-7)` y operandos float32. Con NEP 50 (numpy 2), `float32 + 1e-7` se queda en float32, pero conviene no depender de eso: se usa un dtype explícito.
- El selector elige con `np.argmin(weighted_child_impurity)`, donde `weighted_child_impurity = wR·G_R + wL·G_L` es la misma clave que sklearn (con el signo cambiado), primera ocurrencia.
- La regla de `min_impurity_decrease` usa la fórmula exacta de sklearn y no `gain`, que redondea distinto.
- `gain = G_t - (wL·G_L + wR·G_R)/W` se guarda igual en `CandidateSplits`, porque es lo que va a usar el selector de la fase 2.
- sklearn **no** corta por "ganancia ≤ 0": sigue mientras haya un corte válido y el nodo no sea puro. Se replica eso.

## Alternativas descartadas

- **Usar `gain` para todo.** Es más limpio, pero los empates y los casos límite de `min_impurity_decrease` se resolverían distinto, y la capa 1 no podría ser exacta.
- **Comparar en float64.** Valores consecutivos en float32 como `[1.0, 1.0000001]` generarían un corte que sklearn no genera.

## Consecuencias

- Hay tests unitarios específicos de la precisión float32 (§7.1).
- Si se sube la versión de sklearn, hay que volver a verificar esta tabla (ver 0005).
