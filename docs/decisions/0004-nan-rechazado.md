# 0004 — `NaN` en `X` da `ValueError`

- **Estado:** Aceptada
- **Spec:** `.claude/specs/01-cart-tree.md` §0 (no-objetivos), §2

## Contexto

sklearn 1.8 acepta `NaN` en `X`: en cada corte aprende a qué hijo mandar los valores faltantes.

## Decisión

`NaN` en `X` da `ValueError`. Es una diferencia deliberada con sklearn, documentada en el README.

## Por qué

Soportar faltantes agrega una segunda elección por corte (a qué hijo van los `NaN`). El selector ponderado de la fase 2 también tendría que sortear sobre esa elección, y eso mezcla dos efectos en los experimentos. Ninguno de los datasets previstos tiene faltantes.

## Alternativas descartadas

- **Copiar el comportamiento de sklearn.** Complica el splitter y la fase 2 sin beneficio para la pregunta de investigación.
- **Imputar internamente.** Esconde una transformación de los datos dentro del estimador.

## Relacionada

`max_depth=0` da `ValueError`. Esto no es una diferencia: sklearn hace lo mismo (`InvalidParameterError`, que hereda de `ValueError`). La validación de hiperparámetros sigue los `_parameter_constraints` de sklearn.
