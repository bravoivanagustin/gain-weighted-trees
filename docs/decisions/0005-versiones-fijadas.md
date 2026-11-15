# 0005 — Python 3.13 para desarrollo y sklearn fijado en `>=1.8,<1.9`

- **Estado:** Aceptada
- **Archivos:** `.python-version`, `pyproject.toml`, `.github/workflows/ci.yml`, `tests/test_smoke.py`

## Decisión

- **Python:** `requires-python = ">=3.12"`, desarrollo en 3.13 (fijado en `.python-version`). El CI corre la matriz 3.12 y 3.13.
- **scikit-learn:** `>=1.8,<1.9`. Es un rango angosto a propósito. `tests/test_smoke.py` falla si la versión instalada no es 1.8.
- **numpy:** `>=2` (NEP 50 aplica; ver 0003).
- **Herramientas:** uv (build backend `uv_build`), ruff, pytest, hypothesis, pytest-cov.

## Por qué sklearn angosto

El árbol reproduce semánticas internas de sklearn (0003): umbrales, comparaciones en float32, orden de operaciones de la mejora. Esas cosas pueden cambiar entre versiones menores sin aviso, porque no son API pública. Un rango amplio dejaría pasar un cambio que rompe la verificación sin que se note por qué.

## Cómo subir de versión

1. Releer `sklearn/tree/_splitter.pyx`, `_partitioner.pxd`, `_criterion.pyx`, `_tree.pyx` y `_classes.py` de la versión nueva y comparar con la tabla de 0003.
2. Volver a pasar `tests/test_equivalence_sklearn.py` completo.
3. Actualizar el rango, `test_smoke.py` y la tabla de 0003, y escribir una decisión nueva si algo cambió.
