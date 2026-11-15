# SPEC — Árbol de decisión CART en numpy (fase 1 de "cortes sorteados por ganancia")

> Nombre de trabajo del paquete: `gwtree` (gain-weighted trees). Renombrar libremente.

## 0. Contexto y objetivo

El proyecto final busca **reducir la varianza de Random Forest sorteando el corte de cada nodo con probabilidad ponderada por su ganancia**, en lugar de elegir siempre el mejor. Para poder afirmar algo sobre eso, primero hace falta un **árbol clásico (CART) propio**, rápido y verificado contra scikit-learn. Sobre esa base se va a enchufar la selección aleatoria.

**Objetivo de esta fase:** implementar un `DecisionTreeClassifier` en numpy puro que:

1. elija en cada nodo un corte de **ganancia máxima** y aplique las **mismas reglas de parada** que `sklearn.tree.DecisionTreeClassifier(criterion="gini", splitter="best")`; con un solo atributo, el árbol es **idéntico** al de sklearn (verificación en capas, §7.2);
2. tenga la **selección del corte desacoplada** del cálculo de los candidatos (punto de extensión para la fase 2);
3. sea lo bastante rápido para entrenar cientos de árboles: breast_cancer (569×30) en < 0.2 s por árbol.

### No-objetivos (fase 1)
- Regresión, multi-output, `sample_weight` **no enteros**, `min_weight_fraction_leaf`, `class_weight`, valores faltantes (NaN) y atributos categóricos nativos (se codifican con one-hot fuera del modelo).
  - **`sample_weight` solo acepta enteros ≥ 0** (§2). Es lo que necesita el bootstrap del bosque (§6), y con pesos enteros los conteos ponderados son exactos en float64, así que el razonamiento de empates bit a bit de §5.2 sigue valiendo. Se documenta en "Known differences from scikit-learn" del README.
  - **NaN es una diferencia deliberada con sklearn:** sklearn ≥ 1.3 acepta NaN y aprende hacia qué hijo mandar los faltantes; nosotros tiramos `ValueError` (§2). Soportarlo duplicaría los candidatos (faltantes a la izquierda o a la derecha) y, en la fase 2, obligaría a decidir cómo se pondera esa dirección en el sorteo, una variable más que confundiría la comparación entre métodos. Los datos de los experimentos se imputan fuera del modelo, igual para todos los métodos. Se documenta en "Known differences from scikit-learn" del README.
- Criterios distintos de Gini (entropía queda para después y se agrega como otra función de impureza).
- Ensamble / bootstrap / selección ponderada. Solo se **diseñan los puntos de extensión** (§6).
- Reproducir el RNG interno de sklearn. Solo se igualan las **semánticas** de los empates, no los mismos números aleatorios. Los empates **entre atributos** se resuelven según el orden de `rng.permutation` (§5.5), así que el árbol puede diferir del de sklearn en esos nodos. Eso **no es un bug**: cualquier corte empatado es CART válido. Además, es el límite α → ∞ del selector de la fase 2, que reparte la probabilidad en partes iguales entre los cortes empatados en el máximo. Se documenta en "Known differences from scikit-learn" del README.

---

## 1. Stack y repositorio

- **Python:** `requires-python = ">=3.12"` (lo que el código soporta) y `.python-version` en **3.13** (el intérprete de desarrollo y de los experimentos). Gestionado con **uv** (`uv init`, `uv add`); el entorno exacto queda en `uv.lock`.
- **Dependencias:** `numpy>=2`, `scikit-learn>=1.8,<1.9` (para la API base, la validación de entrada y como oráculo en los tests).
  - El rango de sklearn es **angosto a propósito**: copiamos semánticas internas que cambian entre versiones (por ejemplo, la precisión de `FEATURE_THRESHOLD`, §5.2). Para subir de versión hay que releer los `.pyx` de la nueva y volver a pasar §7.2. Se explica con un comentario junto a la dependencia en `pyproject.toml`.
- **Dependencias de dev:** `pytest`, `pytest-cov`, `hypothesis`, `ruff`.
- **CI:** GitHub Actions con `uv sync --frozen`, `ruff check`, `ruff format --check` y `pytest`, con una matriz de Python 3.12 y 3.13.
- **Convenciones:** identificadores en inglés (como sklearn), docstrings y docs en español o inglés pero consistentes. Type hints en la API pública.

```
arboles/                         # raíz del repo
├── CLAUDE.md
├── .claude/specs/               # una spec por fase: NN-nombre.md
│   └── 01-cart-tree.md          # este archivo
├── pyproject.toml
├── README.md
├── src/gwtree/
│   ├── __init__.py              # exporta DecisionTreeClassifier, export_text
│   ├── _criterion.py            # impureza Gini vectorizada a partir de conteos
│   ├── _splitter.py             # tabla de cortes candidatos de un nodo
│   ├── _selector.py             # protocolo SplitSelector + BestSplitSelector
│   ├── _tree.py                 # estructura de arrays del árbol + builder
│   ├── classifier.py            # DecisionTreeClassifier (API sklearn)
│   └── export.py                # export_text
├── tests/
│   ├── test_criterion.py
│   ├── test_splitter.py
│   ├── test_selector.py
│   ├── test_tree_builder.py
│   ├── test_classifier_api.py
│   └── test_equivalence_sklearn.py
├── benchmarks/
│   └── bench_fit.py
└── experiments/                 # vacío en fase 1
```

---

## 2. API pública

```python
class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_impurity_decrease: float = 0.0,
        max_features: int | float | Literal["sqrt", "log2"] | None = None,
        random_state: int | np.random.Generator | None = None,
        selector: SplitSelector | None = None,   # None -> BestSplitSelector()
    ): ...

    def fit(self, X, y, sample_weight=None) -> Self   # pesos enteros >= 0
    def predict(self, X) -> np.ndarray            # etiquetas originales
    def predict_proba(self, X) -> np.ndarray      # (n_samples, n_classes)
    def apply(self, X) -> np.ndarray              # id de hoja por muestra
    def get_depth(self) -> int
    def get_n_leaves(self) -> int

    # Atributos post-fit
    classes_: np.ndarray
    n_classes_: int
    n_features_in_: int
    feature_names_in_: np.ndarray                 # si X era DataFrame
    tree_: Tree                                    # §3
    feature_importances_: np.ndarray              # property, §5.6
```

- Hereda de `sklearn.base.BaseEstimator` y `ClassifierMixin`. Así funcionan `clone`, `get_params`, `cross_val_score`, `GridSearchCV` y `score`.
- `__init__` **solo guarda** los parámetros (regla de sklearn). Toda la validación ocurre en `fit`.
- `min_samples_split` / `min_samples_leaf` como float significan fracción, con la misma semántica que sklearn (`_classes.py`): `min_samples_leaf = ceil(frac * n_samples)` y `min_samples_split = max(2, ceil(frac * n_samples))`, donde `n_samples = X.shape[0]` **incluye las filas con peso 0**.
- `export_text(tree, feature_names=None, decimals=2) -> str`, con un formato parecido al de `sklearn.tree.export_text`.

### Validación de entrada (en `fit`)
- `X` se convierte a `np.float32` contiguo, como hace sklearn. **Esto es necesario para igualar los umbrales** (§5.3).
- `NaN` o `inf` en `X` → `ValueError` (diferencia deliberada con sklearn, ver no-objetivos en §0).
- **Hiperparámetros:** mismas restricciones que `DecisionTreeClassifier._parameter_constraints` de sklearn; un valor fuera de rango → `ValueError`. Por ejemplo, `max_depth` tiene que ser un entero ≥ 1 o `None`: `max_depth=0` es un error, igual que en sklearn.
- `y` 1-D de cualquier tipo de etiqueta: `self.classes_, y_enc = np.unique(y, return_inverse=True)`. Internamente se trabaja con `y_enc` (`intp`).
- `sample_weight`: `None` equivale a todos 1. Si se pasa, tiene que ser 1-D, de largo `n_samples`, con valores **enteros ≥ 0** (se acepta dtype float con valores enteros, como el `bincount` del bosque) y al menos uno positivo; si no, `ValueError`. Internamente es `float64`.
  - Igual que sklearn (`_splitter.pyx`, `init`), las filas con peso 0 **no entran al árbol**: no cuentan en `n_node_samples` ni para `min_samples_*`.
- En `predict`/`predict_proba`/`apply`, `X` debe tener `n_features_in_` columnas; si no, `ValueError`.
- Para validar se puede usar `sklearn.utils.validation.validate_data` y `check_is_fitted`.

---

## 3. Representación del árbol (`_tree.py`)

Arrays paralelos indexados por id de nodo, igual que `sklearn.tree._tree.Tree`. Así se pueden comparar campo por campo.

```python
@dataclass
class Tree:
    children_left: np.ndarray    # intp, -1 en hojas
    children_right: np.ndarray   # intp, -1 en hojas
    feature: np.ndarray          # intp, -2 en hojas
    threshold: np.ndarray        # float64, -2.0 en hojas
    impurity: np.ndarray         # float64, Gini del nodo (con conteos ponderados)
    n_node_samples: np.ndarray   # intp, filas distintas con peso > 0
    weighted_n_node_samples: np.ndarray  # float64, suma de pesos del nodo
    value: np.ndarray            # float64 (n_nodes, n_classes): FRACCIONES de clase ponderadas
    max_depth: int
    node_count: int
```

- **Orden de los nodos: preorden, izquierda primero.** Esto coincide con el `DepthFirstTreeBuilder` de sklearn, que apila primero el hijo derecho y después el izquierdo.
- **Convención del corte:** `x[feature] <= threshold` → izquierda; si no, derecha.
- Se construye con listas que crecen y se convierten a arrays al final, o con arrays prealocados que se agrandan al doble. **Sin recursión de Python**: se usa una pila explícita, para no depender del límite de recursión.
- `value` guarda **fracciones** de los conteos ponderados (`wcounts / wcounts.sum()`). Verificado: `sklearn.tree_.value` en 1.8 también guarda fracciones, así que se compara directo.

---

## 4. Diseño por capas y punto de extensión

```
classifier.fit
   └── builder (pila depth-first, criterios de parada)        _tree.py
         ├── splitter.candidate_splits(node)  -> CandidateSplits   _splitter.py
         └── selector.select(candidates, rng) -> índice | None     _selector.py
```

### 4.1 `CandidateSplits` (salida del splitter)
Tabla con **todos** los cortes válidos del nodo, en orden canónico: atributos en el orden en que se sortearon (§5.5) y, dentro de cada atributo, umbrales ascendentes.

```python
@dataclass(frozen=True)
class CandidateSplits:
    feature: np.ndarray      # intp   (m,)
    threshold: np.ndarray    # float64 (m,)
    gain: np.ndarray         # float64 (m,)  ganancia LOCAL: G_t - (wL·G_L + wR·G_R) / W
    weighted_child_impurity: np.ndarray  # float64 (m,)  wR·G_R + wL·G_L, en ese orden (§5.2)
    n_left: np.ndarray       # intp   (m,)  filas a la izquierda
    weight_left: np.ndarray  # float64 (m,) suma de pesos a la izquierda
    impurity_left: np.ndarray
    impurity_right: np.ndarray
```

- `gain` es lo que usan los selectores aleatorios (fase 2).
- `weighted_child_impurity` es la clave exacta con la que sklearn compara cortes (§4.2). Mismo argmin que `-gain` en aritmética exacta, pero sin el redondeo de la resta y la división.

### 4.2 `SplitSelector` (protocolo)
```python
class SplitSelector(Protocol):
    def select(self, candidates: CandidateSplits, rng: np.random.Generator) -> int | None:
        """Índice del corte elegido, o None para convertir el nodo en hoja."""
```

- **`BestSplitSelector`** (fase 1): `np.argmin(candidates.weighted_child_impurity)`. sklearn maximiza el proxy `-wR·G_R - wL·G_L` (`_criterion.pyx`, `proxy_impurity_improvement`), que es exactamente el negativo de esta clave. Usar `gain` podría elegir otro corte cuando dos empatan matemáticamente pero redondean distinto. `argmin` devuelve la **primera** ocurrencia del mínimo en orden canónico, lo que reproduce el `>` estricto de sklearn. Consecuencias:
  - los empates **dentro** de un atributo se resuelven igual que en sklearn: gana el umbral menor;
  - los empates **entre** atributos se resuelven por nuestro orden aleatorio de atributos (§5.5), que no es el de sklearn (ver no-objetivos, §0).

  Si `m == 0` → `None`.
- **Fase 2** (no implementar ahora): `GainWeightedSelector(alpha)`, con `p ∝ gain^alpha` sobre los cortes con `gain > 0`, y una variante en dos etapas (atributo → umbral). Se agrega solo implementando este protocolo, sin tocar el builder.

---

## 5. Algoritmo (debe reproducir sklearn 1.8)

Referencias en el código de sklearn: `sklearn/tree/_splitter.pyx` (`node_split_best`), `_partitioner.pyx`, `_criterion.pyx` (`Gini`), `_tree.pyx` (`DepthFirstTreeBuilder`, `compute_feature_importances`).

### 5.1 Impureza (`_criterion.py`)
- `gini(wcounts) = 1 - Σ_k c_k² / W²`, con `c_k` los conteos **ponderados** de clase y `W = Σ_k c_k`. Vectorizada: recibe `wcounts` de forma `(..., n_classes)` y devuelve `(...)`.
- Es la misma cuenta y en el mismo orden que sklearn (`_criterion.pyx`, `Gini`): primero `Σc²`, después `1 - Σc² / (W·W)`. **No** usar `1 - Σ(c/W)²` ni `Σp(1-p)`: son iguales matemáticamente pero redondean distinto.
- Con pesos enteros, `c_k`, `Σc²` y `W·W` son enteros exactos en float64 (mientras `W < 2²⁶`), así que el único redondeo es la división.
- Con `W == 0` no se llama nunca: el splitter no genera ramas vacías.

### 5.2 Cortes candidatos de un nodo, para un atributo `j` (`_splitter.py`)
Entrada: los índices de las filas del nodo `idx` (n filas, todas con peso > 0), `X[idx, j]` (float32), `y_enc[idx]` y `w[idx]` (pesos, float64 con valores enteros).

1. `order = np.argsort(xj, kind="stable")`, `xs = xj[order]`, `ys = y_enc[order]`, `ws = w[idx][order]`.
2. **Atributo constante:** si `xs[-1] <= xs[0] + FEATURE_THRESHOLD`, no genera candidatos y se marca como constante (§5.5).
   - `FEATURE_THRESHOLD = np.float32(1e-7)` y la suma y la comparación se hacen **en float32**, igual que sklearn 1.8 (`_partitioner.pxd`: `cdef const float32_t FEATURE_THRESHOLD = 1e-7`, con `feature_values` en `float32_t`).
   - El dtype va **explícito**, sin depender de las reglas de promoción de numpy (NEP 50): en float32, `1.0` y `1.0000001` (valores consecutivos) se consideran iguales; en float64, no.
3. Conteos ponderados acumulados por la izquierda: `left = np.cumsum(onehot(ys) * ws[:, None], axis=0)[:-1]`, de forma `(n-1, K)`. La fila `i` corresponde a dejar `xs[0..i]` a la izquierda. `right = total - left`. `wL = left.sum(axis=1)`, `wR = W - wL`. Con pesos enteros, todo es exacto.
4. **Posiciones válidas:** `i` tal que `xs[i+1] > xs[i] + FEATURE_THRESHOLD` (en float32, como en el paso 2), `n_left = i+1 >= min_samples_leaf` y `n_right = n-(i+1) >= min_samples_leaf`. `min_samples_leaf` cuenta **filas**, no pesos, igual que sklearn (`_splitter.pyx`: `n_left = current_split.pos - start`).
5. Impureza de los hijos con `gini` (§5.1) sobre `left` y `right`. Después, dos cantidades:
   - `weighted_child_impurity = wR·G_R + wL·G_L`, en ese orden de suma, como el proxy de sklearn (`-wR·G_R - wL·G_L`). Es la clave de `BestSplitSelector`.
   - `gain = G_t - weighted_child_impurity / W`, para los selectores de la fase 2 y para los tests.

   Dos cortes con los mismos conteos dan las mismas dos cantidades **bit a bit**.
6. **Umbral** (punto medio, como sklearn): `thr = np.float64(xs[i]) / 2.0 + np.float64(xs[i+1]) / 2.0`. Si `thr == xs[i+1]` o `thr` es ±inf, entonces `thr = np.float64(xs[i])`.
7. Se concatenan los candidatos de todos los atributos visitados, en el orden de visita.

Complejidad por nodo: `O(d · n log n)`, todo vectorizado por atributo y sin bucles de Python sobre las muestras.

> Nota: dos cortes con conteos distintos pero la misma ganancia matemática pueden diferir en el último bit. `BestSplitSelector` los resuelve igual que sklearn porque compara la misma clave con la misma aritmética (§4.2). En los tests que comparan ganancias entre implementaciones (oráculo de §7.2), dos cortes se consideran **empatados** si sus ganancias difieren en menos de `atol = 1e-12`.

### 5.3 Partición
`mask = X[idx, f] <= thr` (float32 comparado con float64, igual que la predicción). Los hijos se quedan con `idx[mask]` e `idx[~mask]`, conservando el orden relativo.

### 5.4 Builder y criterios de parada (`_tree.py`)
Pila de `(idx, depth, parent, is_left)`. Para cada nodo extraído:

0. La raíz arranca con las filas de peso > 0. `W_root` es la suma de **todos** los pesos (las filas con peso 0 suman 0, así que es lo mismo).
1. Calcular los conteos ponderados `wcounts`, `n = len(idx)` (filas), `W = wcounts.sum()` y `G_t = gini(wcounts)`. Registrar el nodo (preorden), con `n_node_samples = n` y `weighted_n_node_samples = W`, y enlazarlo con su padre.
2. **Hoja antes de buscar** si se cumple alguna de estas condiciones:
   - `max_depth is not None and depth >= max_depth`
   - `n < min_samples_split`
   - `n < 2 * min_samples_leaf`
   - `G_t <= EPSILON`, donde `EPSILON = np.finfo(np.float64).eps` (nodo puro).
3. `cands = splitter(...)`; `k = selector.select(cands, rng)`. Si `k is None` → hoja.
4. **Mejora ponderada**, solo para el corte elegido y con la misma fórmula y orden de operaciones que sklearn (`_criterion.pyx`, `impurity_improvement`):
   `improvement = (W / W_root) * (G_t - (wR / W) * G_R - (wL / W) * G_L)`.
   Si `improvement + EPSILON < min_impurity_decrease` → hoja. No se usa `gain` acá: su redondeo es distinto y cambiaría los casos que quedan justo en el límite.
5. Si no es hoja: guardar `feature` y `threshold`, partir y **apilar primero el derecho y después el izquierdo**.

**Cuidado con la versión de libro de texto:** muchas implementaciones didácticas de CART paran cuando la mejor ganancia es ≤ 0. sklearn **no** lo hace. Sigue mientras haya un corte válido y el nodo no sea puro, aunque la mejor ganancia sea 0. Hay que replicar el comportamiento de sklearn.

### 5.5 `max_features` y orden de atributos
- `max_features` se resuelve así: `None` → `d`; `"sqrt"` → `max(1, int(sqrt(d)))`; `"log2"` → `max(1, int(log2(d)))`; int → ese valor; float → `max(1, int(f * d))`.
- En cada nodo los atributos se recorren en **orden aleatorio** (`rng.permutation(d)`), **incluso con `max_features=None`**, igual que sklearn. Esto define cómo se rompen los empates.
- Se visitan atributos hasta haber **sorteado `max_features`**. Los constantes también cuentan como sorteados, pero si todos los sorteados hasta ese momento son constantes, se sigue sorteando hasta encontrar uno no constante o agotar los atributos.
- *(Optimización opcional)* Recordar los atributos constantes de los ancestros: si un atributo es constante en un nodo, también lo es en sus descendientes.
- `rng = np.random.default_rng(random_state)` una sola vez por `fit`. Con el mismo `random_state`, el árbol tiene que ser bit-idéntico entre corridas.

### 5.6 `feature_importances_`
Para cada nodo interno `t` con hijos `L` y `R`:
`imp[feature_t] += W_t·G_t − W_L·G_L − W_R·G_R` (con `weighted_n_node_samples`, como `_tree.pyx`, `compute_feature_importances`).
Después se divide por `W_root` y se normaliza para que la suma sea 1 (si la suma es > 0; si no, queda en ceros). Debe coincidir con `sklearn ... .feature_importances_`.

### 5.7 Predicción
- `apply`: recorrido **vectorizado por niveles**. Se mantiene un vector `node` de forma `(n_samples,)` y se itera mientras haya muestras en nodos internos (a lo sumo `max_depth` iteraciones). No hay bucles de Python por muestra.
- `predict_proba = tree_.value[apply(X)]`.
- `predict = classes_[argmax(proba, axis=1)]`. Ante empate gana la clase de menor índice, igual que sklearn.

---

## 6. Hooks pensados para la fase 2 (solo diseño)

- `selector` es un parámetro del estimador → `GainWeightedSelector(alpha=...)` se enchufa sin tocar nada más.
- El builder recibe `rng` y lo pasa al selector → un solo stream aleatorio por árbol, reproducible.
- El splitter devuelve **todos** los candidatos (no solo el mejor) → la fase 2 puede sortear sobre ellos.
- Más adelante, `RandomForest` = bootstrap + `max_features` + selector. El bootstrap se pasa como **pesos**, igual que sklearn (`ensemble/_forest.py`, `_parallel_build_trees`): `sample_weight = np.bincount(indices, minlength=n_samples)`. **No** como filas repetidas: con filas repetidas, las copias contarían para `min_samples_leaf` y los árboles diferirían de los de `RandomForestClassifier`, que es el control de la fase 4. Así, el árbol no copia `X` y las filas fuera de la bolsa (peso 0) quedan fuera del árbol.

---

## 7. Tests (pytest)

### 7.1 Unitarios
- **`test_criterion`:** Gini de un nodo puro = 0; de 2 clases 50/50 = 0.5; de K clases uniformes = 1 − 1/K. Coincide con `Σp(1-p)` con `atol = 1e-15`. Funciona sobre arrays batch y con conteos ponderados.
- **`test_splitter`:** comparar contra una **búsqueda por fuerza bruta** (probar cada valor único con un `<=` y calcular la ganancia desde cero con un bucle). Las ganancias y los umbrales tienen que coincidir. Casos borde:
  - todos los valores iguales → sin candidatos;
  - valores repetidos → no hay umbral entre valores iguales;
  - `min_samples_leaf` filtra bien;
  - valores que difieren en menos de 1e-7 se tratan como iguales;
  - precisión float32: `[1.0, 1.0000001]` (consecutivos en float32) con clases distintas **no** genera candidato, y dos valores apenas por encima del umbral **sí** lo generan;
  - umbral exactamente en el punto medio en float32/float64;
  - con pesos: la fuerza bruta usa conteos ponderados, y `min_samples_leaf` cuenta filas (una fila con peso 5 es 1 fila).
- **`test_selector`:** `BestSplitSelector` devuelve la primera ocurrencia del mínimo de `weighted_child_impurity`; devuelve `None` con 0 candidatos.
- **`test_tree_builder`:**
  - `y` de una sola clase → 1 hoja;
  - `n=1`;
  - `max_depth=0` → `ValueError`, igual que sklearn;
  - `max_depth=1` → a lo sumo 3 nodos;
  - preorden: `children_left[i] == i+1` en todo nodo interno;
  - ninguna hoja con menos de `min_samples_leaf` muestras.
  - Pesos:
    - `sample_weight` no entero, negativo, de largo equivocado o todo en cero → `ValueError`;
    - con pesos en `{0, 1}`, el árbol es igual al entrenado solo con las filas de peso 1 (con `min_samples_*` enteros; con fracciones, sklearn las resuelve contra `X.shape[0]` y cambian a propósito);
    - `sample_weight = np.ones(n)` da el mismo árbol que `None`.
  - Con `hypothesis`, para datos y pesos enteros aleatorios:
    - la suma de `n_node_samples` y de `weighted_n_node_samples` de los hijos es igual a la del padre;
    - el árbol completo sin restricciones tiene accuracy de train = 1 (sobre las filas con peso > 0) **si no hay dos filas con distinta `y` que sean indistinguibles**: iguales después de convertir a float32, o a distancia ≤ `FEATURE_THRESHOLD` en float32 en todos los atributos. La estrategia de hypothesis tiene que filtrar esos casos (o el test tiene que excluirlos), porque si no los genera y la propiedad es falsa.

### 7.2 Corrección contra sklearn (`test_equivalence_sklearn.py`) — el test más importante

**Por qué en capas.** La ganancia de Gini depende solo de los conteos de clase a cada lado del corte, no de los valores de `x`. Por eso, aunque `X` sea continua, los empates de ganancia **entre atributos** son muy frecuentes en los nodos chicos. Por ejemplo, en un nodo con 2 muestras de cada clase y `d = 20`, se esperan unos 6–7 atributos que separan perfecto. sklearn y nosotros desempatamos con RNG distintos (§0), así que con `d > 1` los árboles completos casi nunca son idénticos, aunque los dos sean CART correctos. La verificación se arma en capas:

| Capa | Datos | Garantiza |
|---|---|---|
| 1. Igualdad exacta | sintéticos, `d = 1` | umbrales, reglas de parada, preorden, `value`, importancias, predicción |
| 2. Oráculo nodo por nodo | sintéticos con `d > 1` y reales | cada nodo elige un corte de ganancia máxima y cada hoja para por la regla correcta |
| 3. Distribución | reales, `max_features < d` | comportamiento estadístico igual al de sklearn |

En todas las capas, sklearn se usa como `sklearn.tree.DecisionTreeClassifier(criterion="gini", random_state=0, ...)` con los mismos hiperparámetros.

**Grilla de hiperparámetros** (capas 1 y 2):

```
max_depth ∈ {None, 1, 3, 8}
min_samples_split ∈ {2, 10, 0.05}
min_samples_leaf ∈ {1, 5, 0.02}
min_impurity_decrease ∈ {0.0, 0.01}
```

**Datos sintéticos:** `X = rng.normal(size=(n, d))`, con `y` generada por una regla no lineal con ruido. Se prueba con `n ∈ {50, 300, 1000}`, `K ∈ {2, 3, 5}` y varias semillas.

**Pesos** (capas 1 y 2): cada caso se corre sin pesos y con pesos de bootstrap, `sample_weight = np.bincount(rng.integers(0, n, n), minlength=n)` (incluye ceros), pasando los mismos pesos a sklearn con `fit(X, y, sample_weight=w)`.

#### Capa 1 — igualdad exacta con `d = 1`
Con un solo atributo no hay empates entre atributos, y los empates dentro del atributo se resuelven igual que en sklearn (§4.2). Con `max_features=None`, se compara campo a campo:
- `node_count`, `children_left`, `children_right`, `feature`, `n_node_samples` y `weighted_n_node_samples`: **iguales**.
- `threshold`, `impurity` y `value`: `np.allclose(rtol=0, atol=1e-12)`.
- `feature_importances_`, `predict_proba(X_test)` y `predict(X_test)`: iguales o `allclose`.

#### Capa 2 — oráculo nodo por nodo
Datos sintéticos con `d ∈ {5, 20}` y datos reales (`load_iris`, `load_breast_cancer`, `load_wine`), con `max_features=None`.

Un helper de test reconstruye las muestras de cada nodo recorriendo nuestro árbol sobre `X_train`.

Para cada **nodo interno**, con las muestras del nodo:
1. Se entrena un stump de sklearn: `DecisionTreeClassifier(max_depth=1, min_samples_leaf=<valor absoluto resuelto contra la raíz>)`, con `sample_weight` = los pesos de esas filas.
2. Nuestra ganancia es igual a la del stump, con `atol = 1e-12`.
3. El `(feature, threshold)` del stump está en **nuestro conjunto de cortes empatados en el máximo**, recalculado con nuestro splitter (empate = diferencia de ganancia < `1e-12`, §5.2). Ejemplo: en iris, `petal length <= 2.45` y `petal width <= 0.8` separan exactamente a setosa y empatan, así que cualquiera de los dos es válido.
4. `impurity` y `value` del nodo coinciden con los del stump en su raíz.

Para cada **hoja** que no se detuvo por las reglas previas a la búsqueda (`max_depth`, `min_samples_split`, `2·min_samples_leaf`, nodo puro; §5.4), se verifica una de dos cosas: el stump no corta, o la `improvement` de su corte (§5.4, paso 4, con `W_root` del árbol completo) cumple `improvement + EPSILON < min_impurity_decrease`.

Además, con el árbol completo en datos reales, la **accuracy de train es igual** a la de sklearn (= 1.0 si no hay filas `X` duplicadas con distinta `y`).

Las importancias no se comparan en esta capa, porque los empates las cambian legítimamente.

#### Capa 3 — distribución con `max_features < d`
No se pueden comparar árboles individuales (el RNG es distinto). Se compara la **distribución**: la accuracy media sobre 30 semillas tiene que estar dentro de ±2 desvíos estándar de la de sklearn.

### 7.3 API
- `sklearn.base.clone` funciona; `get_params`/`set_params` funcionan.
- `cross_val_score(DecisionTreeClassifier(), X, y)` corre.
- Las etiquetas string y las no contiguas (`[3, 7, 7, 3]`) se devuelven tal cual.
- Llamar `predict` antes de `fit` → `NotFittedError`.
- Un número equivocado de columnas → `ValueError`.
- Un DataFrame de entrada guarda `feature_names_in_`.
- *(Extra)* Pasar `sklearn.utils.estimator_checks.parametrize_with_checks`, o documentar qué checks se saltean y por qué.

### 7.4 Cobertura
≥ 90 % de líneas en `src/gwtree`.

---

## 8. Performance (`benchmarks/bench_fit.py`)

Mide el tiempo de `fit` y `predict` (mediana de 5 corridas) contra sklearn e imprime una tabla.

| Dataset | Objetivo `fit` |
|---|---|
| breast_cancer (569×30) | < 0.2 s |
| `make_classification(10_000, 20)` | < 5 s |
| `predict` de 100k filas | < 0.5 s |

Si no se alcanza: perfilar (`cProfile`) antes de optimizar. Las opciones, en este orden:
1. evitar copias de `X[idx]` (indexar por columna: `X[idx, j]`);
2. preordenar `X` una vez por atributo y propagar los órdenes a los hijos;
3. `numba` (solo si hace falta; queda como dependencia opcional).

---

## 9. Hitos y criterio de "terminado"

| Hito | Entregable | Hecho cuando |
|---|---|---|
| M0 | Repo, `pyproject`, CI, ruff, esqueleto de paquetes | CI verde con un test vacío |
| M1 | `_criterion` + `_splitter` + `_selector`, con pesos | §7.1 en verde, incluido fuerza bruta |
| M2 | Builder + `fit` (con `sample_weight`)/`predict`/`predict_proba`/`apply` | §7.1 builder + §7.3 en verde |
| M3 | Corrección contra sklearn | §7.2 en verde (capas 1, 2 y 3) |
| M4 | `feature_importances_`, `export_text`, `get_depth`, `get_n_leaves` | iguales a sklearn en la capa 1 de §7.2 |
| M5 | Benchmarks | objetivos de §8 cumplidos o desvío documentado |

**Definición de terminado (fase 1):**
- todo lo anterior en verde;
- `README` con un ejemplo de uso y la tabla de equivalencia/benchmark;
- ninguna dependencia de pandas en `src/`.

---

## 10. Instrucciones para quien implemente (humano o agente)

- Trabajar **por hitos y en orden**. No avanzar si los tests del hito anterior no pasan.
- Ante cualquier duda de semántica, **la referencia es el código de sklearn** (`sklearn/tree/*.pyx` de la versión instalada), no la documentación ni este spec. Si el spec contradice al código de sklearn, gana sklearn y se corrige el spec.
- No optimizar antes de M3: primero correcto, después rápido.
- Cada diferencia conocida con sklearn va a una sección "Known differences from scikit-learn" del README, con su causa.
