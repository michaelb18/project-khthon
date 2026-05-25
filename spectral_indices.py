import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _norm_diff(a, b):
    return (a - b) / (a + b + 1e-10)


def _flatten(X):
    return X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X


def _output_shape(X, cols):
    if X.ndim == 3:
        return X.shape[:2] + (cols,) if cols > 1 else X.shape[:2]
    return X


def _reshape_result(X, result, cols):
    return result.reshape(X.shape[0], X.shape[1]) if X.ndim == 3 and cols == 1 else result


class NDVI(BaseEstimator, TransformerMixin):
    """(NIR - Red) / (NIR + Red)"""
    def __init__(self, nir=7, red=3):
        self.nir = nir
        self.red = red

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = _norm_diff(X[..., self.nir], X[..., self.red])
        return _reshape_result(X, result, 1)


class NBR(BaseEstimator, TransformerMixin):
    """(NIR - SWIR2) / (NIR + SWIR2)"""
    def __init__(self, nir=7, swir2=11):
        self.nir = nir
        self.swir2 = swir2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = _norm_diff(X[..., self.nir], X[..., self.swir2])
        return _reshape_result(X, result, 1)


class NDBI(BaseEstimator, TransformerMixin):
    """(SWIR1 - NIR) / (SWIR1 + NIR)"""
    def __init__(self, swir1=10, nir=7):
        self.swir1 = swir1
        self.nir = nir

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = _norm_diff(X[..., self.swir1], X[..., self.nir])
        return _reshape_result(X, result, 1)


class NDWI(BaseEstimator, TransformerMixin):
    """(Green - NIR) / (Green + NIR)"""
    def __init__(self, green=2, nir=7):
        self.green = green
        self.nir = nir

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = _norm_diff(X[..., self.green], X[..., self.nir])
        return _reshape_result(X, result, 1)


class BSI(BaseEstimator, TransformerMixin):
    """((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))"""
    def __init__(self, swir1=10, red=3, nir=7, blue=1):
        self.swir1 = swir1
        self.red = red
        self.nir = nir
        self.blue = blue

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        num = (X[..., self.swir1] + X[..., self.red]) - (X[..., self.nir] + X[..., self.blue])
        den = (X[..., self.swir1] + X[..., self.red]) + (X[..., self.nir] + X[..., self.blue])
        result = num / (den + 1e-10)
        return _reshape_result(X, result, 1)


class EVI(BaseEstimator, TransformerMixin):
    """2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
    def __init__(self, nir=7, red=3, blue=1):
        self.nir = nir
        self.red = red
        self.blue = blue

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nir = X[..., self.nir]
        red = X[..., self.red]
        blue = X[..., self.blue]
        result = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1e-10)
        return _reshape_result(X, result, 1)


INDEX_MAP = {
    "NDVI": NDVI,
    "NBR": NBR,
    "NDBI": NDBI,
    "NDWI": NDWI,
    "BSI": BSI,
    "EVI": EVI,
}


class SpectralIndexExtractor(BaseEstimator, TransformerMixin):
    """Compute multiple spectral indices in one transformer.

    Parameters
    ----------
    indices : list of str, optional
        Index names to compute. Defaults to all available.

    Examples
    --------
    >>> pipe = Pipeline([
    ...     ("indices", SpectralIndexExtractor(["NDVI", "NBR", "NDBI"])),
    ...     ("clf", RandomForestClassifier()),
    ... ])
    """
    def __init__(self, indices: list[str] | None = None):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        names = self.indices or list(INDEX_MAP)
        results = []
        for name in names:
            cls = INDEX_MAP.get(name)
            if cls is None:
                raise ValueError(f"Unknown index '{name}'. Available: {list(INDEX_MAP)}")
            t = cls()
            result = t.transform(X)
            results.append(result.ravel())
        stacked = np.column_stack(results)
        if X.ndim == 3:
            return stacked.reshape(X.shape[0], X.shape[1], -1)
        return stacked
