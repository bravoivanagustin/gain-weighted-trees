# CLAUDE.md

## Qué es este proyecto

Investigación sobre **cómo reducir la varianza de Random Forest** cambiando la manera en que cada árbol elige sus cortes.

- **Hipótesis.** CART elige siempre el corte de mayor ganancia. RF agrega azar con `max_features`, pero eso sortea por *ranking* y no tiene en cuenta *cuánto* mejor es un corte que otro. Proponemos **sortear el corte con probabilidad ponderada por su ganancia**, por ejemplo `p ∝ gain^α`. Con α → ∞ se recupera CART; con α = 0 el sorteo es uniforme entre los cortes con ganancia positiva.
- **Mecanismo esperado.** La varianza de un ensamble es aproximadamente `ρσ² + (1−ρ)σ²/B`. Sortear bien los cortes debería bajar la correlación entre árboles (`ρ`) sin subir demasiado el error de cada árbol. **Hay que medir las dos cosas**, no solo la accuracy final.
- **Trabajo relacionado contra el que hay que compararse:**
  - Random Forest (Breiman, 2001)
  - Extremely Randomized Trees (Geurts et al., 2006; `ExtraTreesClassifier` en sklearn)
  - "Random split selection" de Dietterich (2000): elegir uniformemente entre los 20 mejores cortes.

## Fases

1. **[ACTUAL] Árbol CART propio en numpy**, verificado nodo por nodo contra `sklearn.tree.DecisionTreeClassifier`. La especificación completa está en `.claude/specs/01-cart-tree.md`: leerla antes de tocar código.
2. `GainWeightedSelector(alpha)`: selección ponderada, también en una variante de dos etapas (primero el atributo, después el umbral).
3. Bosque propio: bootstrap + `max_features` + selector.
4. Experimentos: sesgo, varianza y `ρ` en función de α, contra RF, ExtraTrees y Dietterich top-k.

No adelantar fases. Una fase empieza cuando la anterior tiene todos sus tests en verde.

Las specs viven en `.claude/specs/`, una por fase, con el formato `NN-nombre.md`, donde `NN` es el número de fase (por ejemplo, `02-gain-weighted-selector.md`). `00-roadmap.md` es el plan maestro.

## Contexto y estado

- **`.claude/STATUS.md`**: qué está resuelto, qué falta y cómo retomar. Leerlo al empezar una sesión y actualizarlo al terminarla.
- **`docs/decisions/`**: decisiones de diseño con sus alternativas descartadas. Antes de cambiar algo que una decisión fija, leerla; si se cambia, escribir una decisión nueva que la reemplace.
- **`docs/open-questions.md`**: preguntas abiertas de las fases 2–4.

## Comandos

```bash
uv sync                                   # instalar dependencias
uv run pytest                             # todos los tests
uv run pytest tests/test_equivalence_sklearn.py -x   # equivalencia con sklearn
uv run ruff check . && uv run ruff format --check .
uv run python benchmarks/bench_fit.py     # tiempos contra sklearn
```

## Arquitectura (detalle en `.claude/specs/01-cart-tree.md` §3–§5)

```
classifier.fit → builder (pila, depth-first, preorden)
                   ├── splitter: TODOS los cortes candidatos del nodo → CandidateSplits
                   └── selector.select(candidates, rng) → índice | None
```

- **`src/gwtree/_splitter.py`**: devuelve **todos** los candidatos con su ganancia, no solo el mejor. Esto es lo que habilita la fase 2; no "optimizarlo" devolviendo solo el máximo.
- **`src/gwtree/_selector.py`**: el único lugar donde se decide qué corte usar. Las variantes nuevas implementan el protocolo `SplitSelector` sin tocar el builder ni el splitter.
- **`src/gwtree/_tree.py`**: arrays paralelos con las mismas convenciones que `sklearn.tree._tree.Tree`, para poder compararlos campo por campo.

## Invariantes (no romper)

- **Corrección contra sklearn (verificación en capas, spec §7.2).** Con `BestSplitSelector` y `max_features=None`:
  1. con un solo atributo, el árbol es **idéntico** al de sklearn campo por campo (estructura, umbrales, `value` e importancias);
  2. con cualquier dato, **cada nodo interno elige un corte de ganancia máxima y cada hoja se detiene por la misma regla que sklearn**, verificado nodo por nodo con stumps de sklearn;
  3. con `max_features < d`, la accuracy coincide **en distribución**.

  Los empates entre atributos se resuelven con nuestro propio RNG: cualquier corte empatado es CART válido y no se intenta copiar el desempate de sklearn. Si un cambio rompe `test_equivalence_sklearn.py`, el cambio está mal.
- **Fuente de verdad.** Ante una duda de semántica (umbrales, empates, criterios de parada), la referencia es el **código** de sklearn de la versión instalada (`.venv/Lib/site-packages/sklearn/tree/*.pyx`), no la documentación ni la memoria.
- **Reproducibilidad.** Toda la aleatoriedad sale de un único `np.random.Generator` creado en `fit` a partir de `random_state`. Prohibido usar `np.random.*` global o `random`. Con la misma semilla, el resultado tiene que ser bit-idéntico.
- **Sin pandas en `src/`.** Internamente `X` es un array `float32`, `y` son enteros `intp` y `sample_weight` es `float64` con valores enteros ≥ 0, como en sklearn.
- **Bootstrap como pesos.** El bosque pasa el bootstrap como `sample_weight = bincount(indices)`, igual que `RandomForestClassifier`, nunca como filas repetidas: las copias contarían para `min_samples_leaf` y romperían el control contra RF.
- **Sin bucles de Python por muestra.** Se vectoriza por atributo con numpy. Primero correcto, después rápido: no optimizar antes de que pase la equivalencia.

## Convenciones

- Python 3.12+ (desarrollo en 3.13, fijado en `.python-version`), uv, ruff (lint y formato), pytest + hypothesis.
- sklearn fijado en `>=1.8,<1.9` a propósito: copiamos semánticas internas que cambian entre versiones. No subirlo sin releer sus `.pyx` y volver a pasar la verificación contra sklearn.
- Los identificadores van en inglés y siguen los nombres de sklearn (`max_features`, `min_samples_leaf`, `tree_`, `classes_`).
- API compatible con sklearn: heredar de `BaseEstimator`/`ClassifierMixin` y que `__init__` solo guarde parámetros.
- Cada diferencia conocida con sklearn se documenta en el README ("Known differences from scikit-learn"), con su causa.

## Reglas para los experimentos (fases 3–4)

- Comparar siempre con los mismos `n_estimators`, `max_features`, datos y particiones para todos los métodos.
- Usar validación cruzada repetida con varias semillas y reportar media ± desvío. **Nunca elegir semillas ni hiperparámetros mirando el test.**
- Medir el mecanismo, no solo el accuracy: error medio de cada árbol, correlación promedio entre árboles y varianza de las predicciones entre bosques entrenados con distintas semillas.
- Los resultados van a `experiments/results/` con la config, la semilla y el commit, para poder regenerar cada número y cada gráfico.
- Antes de afirmar que el método reduce la varianza, controlar que tu bosque con α → ∞ reproduzca a `RandomForestClassifier` en distribución. Si no lo reproduce, las diferencias pueden ser bugs y no el método.
