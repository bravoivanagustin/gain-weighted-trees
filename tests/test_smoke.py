import sklearn

import gwtree


def test_package_imports():
    assert gwtree.__version__


def test_sklearn_in_supported_range():
    # The tree copies internal scikit-learn semantics that change between releases,
    # so running against another version would make the comparison tests meaningless.
    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    assert (major, minor) == (1, 8)
