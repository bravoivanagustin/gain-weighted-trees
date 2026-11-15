# Registro de decisiones

Cada archivo registra una decisión de diseño que tenía alternativas razonables: el contexto, qué se decidió y qué se descartó y por qué. La spec (`.claude/specs/`) dice **qué** hace el código; este registro guarda **por qué no se hizo otra cosa**.

Formato: `NNNN-titulo-corto.md`, numeración correlativa. Una decisión no se borra: si cambia, se escribe una nueva que la reemplaza y se marca la vieja como "Reemplazada por NNNN".

| N.º | Decisión | Estado |
|---|---|---|
| [0001](0001-verificacion-en-capas.md) | Verificar contra sklearn en capas y no copiar su desempate entre atributos | Aceptada |
| [0002](0002-pesos-enteros-y-bootstrap.md) | `sample_weight` entero desde la fase 1; bootstrap como pesos | Aceptada |
| [0003](0003-aritmetica-igual-a-sklearn.md) | Reproducir la aritmética de sklearn (float32, clave del argmin, fórmula de la mejora) | Aceptada |
| [0004](0004-nan-rechazado.md) | `NaN` en `X` da `ValueError` | Aceptada |
| [0005](0005-versiones-fijadas.md) | Python 3.13 para desarrollo y sklearn fijado en `>=1.8,<1.9` | Aceptada |
