import pandas as pd
import pytest

from stringcluster import STOP_TOKENS, StringCluster

TEST_SERIES = pd.Series(
        ['Johnson & Johnson, Inc.', 'Johnson & Johnson Inc.', 'Johnson & Johnson Inc',
         'Johnson & Johnson', 'Intel Corp', 'Intel Corp.', 'Intel Corporation', 'Google',
         'Apple', 'Amazon', 'Amazon Inc', 'Comcast Inc.', 'Comcast Corp']
    )
MASTER = ['Johnson & Johnson', 'Intel Corp', 'Google', 'Apple Inc', 'Amazon', 'Comcast']


class TestStringCluster:
    def test_master_list(self):
        c = StringCluster(ngram_size=2, stop_tokens=STOP_TOKENS)
        actual = c.fit_transform(TEST_SERIES, MASTER)
        expected = pd.Series(
            ['Johnson & Johnson']*4 +
            ['Intel Corp'] * 3 +
            ['Google'] +
            ['Apple Inc'] +
            ['Amazon'] * 2 +
            ['Comcast'] * 2
        )
        pd.testing.assert_series_equal(actual, expected)

    def test_no_master_list(self):
        c = StringCluster(ngram_size=2, stop_tokens=STOP_TOKENS)
        actual = c.fit_transform(TEST_SERIES)
        assert actual.nunique() == 6
