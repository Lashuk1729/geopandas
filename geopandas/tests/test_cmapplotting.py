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
        self.df = GeoDataFrame(read_file(get_path('naturalearth_lowres')))
        self.df = self.df[self.df['continent'] != 'Antarctica']
        self.df = self.df[self.df['continent'] != 'Seven seas (open ocean)']

        cmap = mpl.cm.viridis_r
        self.norm = mpl.colors.BoundaryNorm([4e7, 6e7, 8e7, 1e8, 2e8, 4e8, 1e9, 1.2e9], cmap.N)

    def test_min_max(self):
        ax = self.df.plot(column='pop_est')
        actual_lim = ax.collections[0].get_clim()
        expected_lim = (self.df['pop_est'].min(), self.df['pop_est'].max())
        assert actual_lim == expected_lim

    def test_norm(self):
        ax = self.df.plot(column='pop_est', norm = self.norm)
        actual_norm = ax.collections[0].norm(list(self.df['pop_est']))
        print(actual_norm)

        expected_norm = self.norm(list(self.df['pop_est']))
        print(expected_norm)
        np.testing.assert_array_equal(actual_norm,expected_norm)

    def testdf_min_max(self):
        continent1 = self.df['continent'] == 'Asia'
        continent2 = self.df['continent'] == 'Oceania'

        ax = self.df.plot(column='pop_est', norm = self.norm)
        actual_lim = ax.collections[0].get_clim()

        ax1 = self.df[continent1 & self.df['pop_est']].plot(column='pop_est', norm = self.norm)
        actual_lim1 = ax1.collections[0].get_clim()

        ax2 = self.df[continent2 & self.df['pop_est']].plot(column='pop_est', norm = self.norm)
        actual_lim2 = ax2.collections[0].get_clim()

        assert actual_lim == actual_lim1
        assert actual_lim == actual_lim2
        assert actual_lim1 == actual_lim2

    def testdf_color(self):
        continent1 = self.df['continent'] == 'Asia'
        continent2 = self.df['continent'] == 'Oceania'

        ax = self.df.plot(column='pop_est', norm = self.norm)
        actual_color = ax.collections[0].get_facecolor()

        ax1 = self.df[continent1 & self.df['pop_est']].plot(column='pop_est', norm = self.norm)
        actual_color1 = ax1.collections[0].get_facecolor()

        ax2 = self.df[continent2 & self.df['pop_est']].plot(column='pop_est', norm = self.norm)
        actual_color2 = ax2.collections[0].get_facecolor()

        np.testing.assert_array_equal(actual_color[0], actual_color1[0])
        np.testing.assert_array_equal(actual_color[0], actual_color2[0])
        np.testing.assert_array_equal(actual_color1[0], actual_color2[0])
