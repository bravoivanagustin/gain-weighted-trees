# Preguntas abiertas de diseño (fases 2–4)

Son preguntas que salieron al revisar el proyecto y que **no bloquean la fase 1**, pero hay que resolver antes de escribir las specs de las fases siguientes. Cuando se resuelva una, la decisión va a `docs/decisions/` y acá se marca como resuelta, con el enlace.

## Fase 2 — `GainWeightedSelector(alpha)`

### 1. Sesgo por la cantidad de umbrales de cada atributo
Si el sorteo `p ∝ gain^α` se hace sobre **todos** los cortes candidatos del nodo, un atributo con muchos valores distintos aporta muchos más candidatos que uno binario y se lo elige más seguido, aunque sus cortes no sean mejores. Con α chico el efecto es fuerte.

- **Opciones:** sorteo plano sobre todos los candidatos; dos etapas (primero el atributo, por ejemplo con su mejor ganancia, y después el umbral dentro de ese atributo, que ya está previsto en `CLAUDE.md`); normalizar por la cantidad de candidatos del atributo.
- **A decidir:** cuál es la variante principal y cuáles se reportan como ablación.

### 2. α = 0 no es ExtraTrees
Con α = 0 el sorteo es uniforme entre los cortes con ganancia positiva. ExtraTrees hace otra cosa: por cada atributo sorteado toma **un** umbral uniforme en `[min, max]` del nodo y se queda con el mejor de esos. Hay que decirlo explícitamente en la spec y en los resultados, para que nadie interprete α = 0 como "equivalente a ExtraTrees". ExtraTrees sigue siendo un baseline aparte.

### 3. Qué hacer si todas las ganancias son 0
sklearn sigue cortando con ganancia 0 mientras el nodo no sea puro (ver `docs/decisions/0003-aritmetica-igual-a-sklearn.md`). Con `p ∝ gain^α`, si todas las ganancias son 0 las probabilidades quedan indefinidas (`0/0`).

- **Opciones:** hoja; sorteo uniforme entre todos los candidatos; caer al comportamiento de `BestSplitSelector`.
- **Restricción:** con α → ∞ el selector tiene que reproducir el comportamiento de la fase 1, incluido este caso, porque si no el control contra RF falla.
- **A decidir también:** si los candidatos con ganancia 0 participan del sorteo cuando hay otros con ganancia positiva (con `gain^α` y α > 0 tienen probabilidad 0; con α = 0, depende de la definición).

### 4. Alternativa `p ∝ exp(β · n · gain)`
`gain^α` es invariante a escala en la ganancia pero no tiene en cuenta el tamaño del nodo: en un nodo de 5 muestras, una diferencia de ganancia es mucho menos confiable que en uno de 5000. `exp(β · n · gain)` (softmax con temperatura) pondera por `n`, y `n · gain` es proporcional a la reducción total de impureza.

- **A decidir:** si se implementa como segunda familia o solo `gain^α`. El protocolo `SplitSelector` permite sumarla sin tocar el builder.

### 5. Estabilidad numérica
Con α grande, `gain^α` desborda o se va a 0. Hay que calcular en escala logarítmica (`α · log(gain)` y normalizar con log-sum-exp) y testear que α muy grande coincide con `BestSplitSelector`.

## Fase 4 — experimentos

### 6. Cómo medir la correlación entre árboles (`ρ`)
Hay varias definiciones y no son equivalentes:

- la correlación de Breiman (2001), basada en márgenes: `ρ̄` en la cota `PE* ≤ ρ̄(1 − s²)/s²`, junto con la fuerza `s`;
- la correlación promedio entre las probabilidades predichas por pares de árboles, sobre datos fuera de la bolsa o de test;
- descomposiciones de sesgo y varianza para la pérdida 0-1: Kohavi-Wolpert (1996) y Domingos (2000).

**A decidir:** cuál es la métrica principal (propuesta: la de Breiman, porque es la que conecta directamente con la cota de error de RF) y cuáles se reportan como complemento.

### 7. Presupuesto de cómputo
Depende del tiempo real de `fit` del árbol propio, que se mide en el M5 de la fase 1 (`benchmarks/bench_fit.py`). Con esos números se estima: (cantidad de α) × (cantidad de métodos) × (datasets) × (repeticiones de CV) × (semillas) × `n_estimators`. Si no entra, las opciones son menos datasets, numba (spec 01 §8) o paralelizar por árbol.

**A decidir después del M5.**
