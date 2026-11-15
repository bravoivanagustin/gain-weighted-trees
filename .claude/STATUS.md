# Estado del proyecto (para retomar)

Última actualización: 2026-11-15. Actualizar este archivo al terminar cada sesión de trabajo.

## Dónde está cada cosa

| Documento | Para qué |
|---|---|
| `CLAUDE.md` | Qué es el proyecto, invariantes, convenciones. Leerlo siempre primero |
| `.claude/specs/00-roadmap.md` | Plan maestro de las 4 fases, estado de cada hito, forma de trabajo, riesgos |
| `.claude/specs/01-cart-tree.md` | Spec completa de la fase 1 (la actual) |
| `docs/decisions/` | Decisiones de diseño con sus alternativas descartadas (0001–0005) |
| `docs/open-questions.md` | Preguntas abiertas de las fases 2–4 |
| `README.md` | Cara pública, en inglés, con "Known differences from scikit-learn" |

## Resuelto

- **Verificación contra sklearn en capas** (igualdad exacta con `d = 1`, oráculo nodo por nodo con stumps, distribución con `max_features < d`) y no copiar el desempate entre atributos → `docs/decisions/0001`.
- **`sample_weight` entero** desde la fase 1 y bootstrap como pesos → `0002`.
- **Aritmética igual a sklearn 1.8**, verificada en su código: float32 en `FEATURE_THRESHOLD`, clave del argmin, fórmula de la mejora, reglas de parada → `0003`.
- **`NaN` → `ValueError`** (diferencia deliberada); `max_depth=0` → `ValueError` (igual que sklearn) → `0004`.
- **Versiones:** Python 3.13 (soporte desde 3.12), sklearn `>=1.8,<1.9` → `0005`.
- **Specs** en `.claude/specs/NN-nombre.md`.
- **Spec de la fase 1 revisada y lista para implementar.**
- **Esqueleto del repo (M0):** `pyproject.toml`, `uv.lock`, LICENSE MIT, `CITATION.cff`, README, `.gitignore`, `.gitattributes`, CI (lint + tests en 3.12/3.13), test de humo. **M0 cerrado:** CI en verde en GitHub (repo privado https://github.com/bravoivanagustin/gain-weighted-trees).
- **Menor, sin urgencia:** el CI avisa que `actions/checkout@v4` y `astral-sh/setup-uv@v6` usan Node 20, que está deprecado. Subirlas a versiones con Node 24 cuando convenga.

## Pendiente, en orden

1. **Confirmar la forma de trabajo** de `00-roadmap.md` (sección *propuesta*: tareas chicas, tests primero, un PR por hito). Después quitar la nota de borrador.
2. **Implementar la fase 1:** M1 → M5, según `01-cart-tree.md` §9 y las tareas de `00-roadmap.md`.
3. **Antes de la fase 2:** resolver las preguntas 1–5 de `docs/open-questions.md` y escribir `02-gain-weighted-selector.md`.
4. **Después del M5:** estimar el presupuesto de cómputo de la fase 4 (pregunta 7).

## Cómo retomar en otra computadora

```bash
git clone https://github.com/bravoivanagustin/gain-weighted-trees.git
cd gain-weighted-trees
uv sync               # instala Python 3.13 y las dependencias exactas de uv.lock
uv run pytest         # tiene que dar 2 tests en verde
uv run ruff check . && uv run ruff format --check .
```

Requisitos: `git` y `uv` (≥ 0.10). uv descarga Python 3.13 si no está instalado.

La memoria de Claude Code es local de cada máquina y no viaja con el repo: todo el contexto necesario está en los documentos de la tabla de arriba. En una sesión nueva, alcanza con pedir "leé `.claude/STATUS.md` y seguimos".
