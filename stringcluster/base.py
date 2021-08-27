"""Module for de-duplicating arrays of strings."""
import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

STOP_TOKENS = r'[\W_]+|(corporation$)|(corp.$)|(corp$)|(incorporated$)|(inc.$)|(inc$)|(company$)|(common$)|(com$)'

Data = Union[List, pd.Series, np.ndarray]


class StringCluster(BaseEstimator, TransformerMixin):
    """
    Transformer for de-duplicating an array-like container of strings.

    Attributes
    ----------
    ngram_size: int
        Size of ngrams to use in TfidfVectorizer.
    threshold: float
        Threshold to determine similarities; only samples above this number are flagged as similar.
    stop_tokens: re.Pattern
        RegEx pattern of stop tokens for use in TfidfVectorizer.
    vec: TfidfVectorizer
        Scikit-Learn TfidfVectorizer.
    similarity_: np.ndarray
        Array of
    labels_: np.ndarray


    Methods
    -------
    fit(X: Data, y: Optional[Data] = None)
        Fit the transformer to data.
    transform(X: Data, y: Optional[Data] = None)
        Transform the data.
    fit_transform(X: Data, y: Optional[Data] = None, **fit_params)
        Fit and transform the data.

    """
    def __init__(self, ngram_size: int = 2, threshold: float = 0.8, stop_tokens: str = r'[\W_]+'):
        """
        Instantiate a StringCluster object.

        Parameters
        ----------
        ngram_size: int
            Size of ngrams to use in TfidfVectorizer; default 2.
        threshold: float
            Threshold to determine similarities; default 0.8; must be between [0, 1].
        stop_tokens: re.Pattern
            RegEx pattern of stop tokens for use in TfidfVectorizer; default r'[\W_]+'.
        """
        self.ngram_size = ngram_size
        self.threshold = threshold
        self.stop_tokens = re.compile(stop_tokens)
        self.vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(ngram_size, ngram_size))

    def fit(self, X: Data, y: Optional[Data] = None) -> "StringCluster":
        """
        Fit the transformer to data.

        Parameters
        ----------
        X: Data
            Array like object containing duplicated strings.
        y: Optional[Data]
            Optional array like object containing 'master list' of values to map similar samples to.

        Returns
        -------
        StringCluster
            Self.
        """
        self.similarity_ = self._get_cosine_similarity(X, y)
        self.labels_ = self._get_labels()
        return self

    def transform(self, X: Data, y: Optional[Data] = None) -> pd.Series:
        """
        Transform data.

        Parameters
        ----------
        X: Data
            Array like object containing duplicated strings.
        y: Optional[Data]
            Optional array like object containing 'master list' of values to map similar samples to.

        Returns
        -------
        pd.Series
            Pandas Series of de-duplicated values.
        """
        if not hasattr(self, 'labels_'):
            raise AttributeError(".fit() method must be called before .transform() method.")
        if y:
            return pd.Series(y)[self.labels_].reset_index(drop=True)
        return pd.Series(X)[self.labels_].reset_index(drop=True)

    def fit_transform(self, X: Data, y: Optional[Data] = None, **fit_params) -> pd.Series:
        """
        Fit and transform the data.

        Parameters
        ----------
        X: Data
            Array like object containing duplicated strings.
        y: Optional[Data]
            Optional array like object containing 'master list' of values to map similar samples to.
        fit_params:
            Optional kwargs; for compatibility, only.

        Returns
        -------
        pd.Series
            Pandas Series of de-deduplicated values.
        """
        return self.fit(X, y).transform(X, y)

    def _get_labels(self) -> np.ndarray:
        """
        Get labels based on similarity scores and given threshold.

        Notes
        -----
        Similarity scores greater than the given threshold are replaced with 1 to setup argmax
        method for identifying and grouping similar samples. This causes duplicates to be renamed
        to the first version within the series.  For example, given a series of
        ['Intel Corp', 'Intel', 'Intel Incorporated'], all three will be renamed to first sample--
        i.e. 'Intel Corp'.

        This also helps reduce the number of inter-group versions which should be replaced with a
        single version.

        Returns
        -------
        np.ndarray
            An array of similarity scores. If `y` is given, the array will be shape
            n_samples by len(y); if no `y` is given, array will be shape n_samples by n_samples.
        """
        return np.where(self.similarity_ > self.threshold, 1., self.similarity_).argmax(1)  # type: ignore

    def _get_cosine_similarity(self, X: Data, y: Optional[Data] = None) -> np.ndarray:
        """Get cosine similarity using fitted TfidfVectorizer and Linear Kernel."""
        if y:
            a, b = self._clean_series(X), self._clean_series(y)
        else:
            a, b = self._clean_series(X), self._clean_series(X)
        self.vec.fit(b)
        return linear_kernel(self.vec.transform(a), self.vec.transform(b))  # type: ignore

    def _clean_series(self, X: Data) -> pd.Series:
        """Clean series of string values."""
        return pd.Series(X).apply(self._clean_string)  # type: ignore

    def _clean_string(self, string: str) -> str:
        """Remove stop tokens and strip whitespace."""
        return self.stop_tokens.sub(' ', string.lower()).strip()


def dedupe_companies():
    """Deduplicate a list of publicly traded companies."""
    series = pd.read_csv('../data/companies.csv')['company']
    c = StringCluster(ngram_size=2, stop_tokens=STOP_TOKENS)
    labs = c.fit_transform(series)
    return pd.DataFrame({'actual': series.reset_index(drop=True), 'label': labs})


if __name__ == '__main__':
    import time
    start = time.time()
    res = dedupe_companies()
    stop = time.time()
    print(res)
    print(f'Process took {stop-start} seconds.')
