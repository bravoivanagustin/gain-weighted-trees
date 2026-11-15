# gain-weighted-trees

[![CI](https://github.com/bravoivanagustin/gain-weighted-trees/actions/workflows/ci.yml/badge.svg)](https://github.com/bravoivanagustin/gain-weighted-trees/actions/workflows/ci.yml)

This project aims to reduce the variance of random forests by changing how each tree chooses its splits. Instead of always taking the split with the highest impurity gain, each node samples a split with probability weighted by its gain (for example `p ∝ gain^α`): α → ∞ recovers CART, and α = 0 samples uniformly among the splits with positive gain. The goal is to lower the correlation between trees without raising the error of each tree too much, and to measure both effects against Random Forest, Extremely Randomized Trees and Dietterich's random split selection.

The first step is a CART decision tree written in NumPy and checked node by node against scikit-learn. The implementation is not written yet.

## Installation

```bash
git clone https://github.com/bravoivanagustin/gain-weighted-trees.git
cd gain-weighted-trees
uv sync
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Known differences from scikit-learn

The tree is designed to differ from `sklearn.tree.DecisionTreeClassifier` in these points:

- **Tie-breaking between features.** When several features give the same maximum gain, the split is chosen by this package's own random feature order, not scikit-learn's. Any tied split is a valid CART split, so trees can differ from scikit-learn's at those nodes.
- **Sample weights.** `sample_weight` must be non-negative integers, which is what bootstrap resampling produces. With integer weights the weighted class counts are exact, so ties between splits are resolved exactly as in scikit-learn.
- **Missing values.** `NaN` in `X` raises `ValueError`; scikit-learn routes missing values to a learned child. Supporting them would add a second choice per split that the gain-weighted sampling would also have to weight.

## License

MIT. See [LICENSE](LICENSE).
