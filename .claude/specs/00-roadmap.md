# 00 — Hoja de ruta del proyecto

> **Estado: borrador.** Las secciones marcadas como *propuesta* las discutimos pero el usuario todavía no las confirmó. Confirmarlas o cambiarlas antes de arrancar el M1, y después borrar esta nota.

Este documento es el plan maestro. Cubre las 4 fases con distinto nivel de detalle: la fase 1 está especificada en `01-cart-tree.md`; las fases 2–4 tienen acá solo el objetivo, la condición de entrada y lo que falta decidir. Cada fase tiene su propia spec `NN-nombre.md` antes de empezar a implementarla.

## Fases

| Fase | Objetivo | Spec | Condición para empezar | Estado |
|---|---|---|---|---|
| 1 | Árbol CART propio en numpy, verificado contra sklearn | `01-cart-tree.md` | — | M0 hecho localmente; M1–M5 pendientes |
| 2 | `GainWeightedSelector(alpha)`, plano y en dos etapas | `02-gain-weighted-selector.md` (a escribir) | Fase 1 terminada; preguntas 1–5 de `docs/open-questions.md` resueltas | No empezada |
| 3 | Bosque propio: bootstrap como pesos + `max_features` + selector | `03-forest.md` (a escribir) | Fase 2 terminada | No empezada |
| 4 | Experimentos: sesgo, varianza y `ρ` en función de α, contra RF, ExtraTrees y Dietterich top-k | `04-experiments.md` (a escribir) | Fase 3 terminada y control "α → ∞ reproduce RF" en verde; preguntas 6–7 resueltas | No empezada |

## Fase 1 en detalle

Los hitos y su criterio de "terminado" están en `01-cart-tree.md` §9. Esta tabla agrega el estado y las tareas.

| Hito | Estado | Tareas |
|---|---|---|
| M0 — repo, CI, esqueleto | Hecho localmente. Falta: primer commit, remoto en GitHub y CI en verde allá | — |
| M1 — `_criterion`, `_splitter`, `_selector`, con pesos | Pendiente | 1. tests de `_criterion` (§7.1) → implementar; 2. test de fuerza bruta del splitter y casos borde (float32, repetidos, `min_samples_leaf`, pesos) → implementar `CandidateSplits` y el splitter; 3. tests de `BestSplitSelector` → implementar |
| M2 — builder, `fit`, `predict`, `predict_proba`, `apply` | Pendiente | 1. tests del builder (§7.1), incluidas las propiedades con hypothesis; 2. `_tree.py` (arrays paralelos, §3) y el builder (§5.4); 3. validación de entrada y de `sample_weight` (§2); 4. tests de API (§7.3) → clasificador |
| M3 — corrección contra sklearn | Pendiente | 1. capa 1; 2. helper de muestras por nodo + capa 2; 3. capa 3 |
| M4 — importancias, `export_text`, `get_depth`, `get_n_leaves` | Pendiente | comparar con sklearn en la capa 1 |
| M5 — benchmarks | Pendiente | `benchmarks/bench_fit.py`; si no se cumplen los objetivos de §8, perfilar antes de optimizar |

## Cómo se trabaja (*propuesta*)

1. **Tareas chicas.** Cada tarea es un commit y tiene un criterio de aceptación que se puede verificar: qué tests nombrados pasan y con qué comando.
2. **Tests primero**, sobre todo en M1–M3. Los tests contra sklearn se pueden escribir antes de la implementación porque el oráculo ya existe, y escribirlos obliga a fijar la interfaz de `CandidateSplits` y del builder.
3. **Un hito está terminado** cuando: sus tests están en verde, `ruff check` y `ruff format --check` pasan, el CI está en verde y la spec está actualizada si algo cambió.
4. **Ramas y PRs.** Una rama y un PR por hito, para que el CI corra en cada uno, y una pasada de `/code-review` antes de mergear.
5. **Decisiones.** Cada decisión con alternativas descartadas va a `docs/decisions/` (una entrada por decisión).
6. **Este documento** se actualiza al cerrar cada hito: estado, qué cambió y por qué.

## Riesgos

| Riesgo | Mitigación |
|---|---|
| numpy puro es lento, porque el splitter devuelve **todos** los candidatos y no solo el mejor | Medir en M5 antes de optimizar; opciones en spec 01 §8 (evitar copias, preordenar, numba opcional). No sacrificar "todos los candidatos": es lo que habilita la fase 2 |
| La capa 2 tarda mucho en CI (un stump de sklearn por nodo, toda la grilla) | Marcar la grilla completa como test lento (`pytest -m slow`) y correr un subconjunto en cada push |
| Cambios internos de sklearn al subir de versión | Rango fijado en `>=1.8,<1.9` y procedimiento de actualización en `docs/decisions/0005-versiones-fijadas.md` |
| El control "α → ∞ reproduce RF" falla en la fase 4 | La fase 1 ya replica bootstrap como pesos y reglas de parada (0002, 0003). Si falla, son bugs y no el método |
