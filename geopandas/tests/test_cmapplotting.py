from __future__ import absolute_import, division

import itertools
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from shapely.geometry import Point, MultiPoint

from geopandas import GeoSeries, GeoDataFrame, plotting, read_file
from geopandas.datasets import get_path

import pytest

@pytest.fixture(autouse=True)
def close_figures(request):
    yield
    plt.close('all')

class TestCMapPlotting:

    def setup_method(self):
        self.N = 10
        self.points = GeoSeries(Point(i, i) for i in range(self.N))

        values = np.arange(self.N)*10
        self.df = GeoDataFrame({'geometry': self.points, 'values': values})

        bounds = np.array([-10, 0, 10, 30, 50, 70, 90])
        self.norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    def test_min_max(self):
        ax = self.df.plot(column='values', norm = self.norm)

        expected_lim = (self.df['values'].min(), self.df['values'].max())
        actual_lim = ax.collections[0].get_clim()

        assert expected_lim == actual_lim

    def test_norm(self):
        ax = self.df.plot(column='values', norm = self.norm)

        expected_norm = self.norm(list(self.df['values']))
        actual_norm = ax.collections[0].norm(list(self.df['values']))

        np.testing.assert_array_equal(actual_norm,expected_norm)
