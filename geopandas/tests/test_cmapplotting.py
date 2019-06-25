from __future__ import absolute_import, division

import itertools
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from shapely.geometry import Point

from geopandas import GeoSeries, GeoDataFrame, plotting
from geopandas.tests.test_plotting import _check_colors

import pytest

@pytest.fixture(autouse=True)
def close_figures(request):
    yield
    plt.close('all')

class TestCMapPlotting:
    def setup_method(self):
        # Intializing the dataframe
        self.N = 10
        self.points = GeoSeries(Point(i, i) for i in range(self.N)).buffer(0.5)
        values = np.arange(self.N)
        levels = ['1','1','2','4','5','1','5','4','3','2']
        self.df = GeoDataFrame({'geometry': self.points, 'values': values, 'levels': levels})

    def test_customnorm(self):
        # test color without additional parameters
        ax = self.df.plot(column='values')
        actual_color = ax.collections[0].get_facecolors()
        cmap = ax.collections[0].get_cmap()
        expected_color = cmap(np.arange(self.N) / (self.N - 1))
        np.testing.assert_array_equal(actual_color, expected_color)

        # test color with norm
        cmap = mpl.cm.Greys
        norm = mpl.colors.BoundaryNorm([1,3,6,8], cmap.N)
        ax = self.df.plot(column='values', norm = norm)
        actual_color = ax.collections[0].get_facecolors()
        cmap = ax.collections[0].get_cmap()
        expected_color = cmap(norm(list(self.df['values'])))
        np.testing.assert_array_equal(actual_color, expected_color)

    def test_minmax(self):
        # test colorbar min,max without additional parameters
        ax = self.df.plot(column='values', legend=True)
        colorbar_color = ax.get_figure().axes[1].collections[0].get_clim()
        point_color = ax.collections[0].get_clim()
        assert colorbar_color == point_color

        # test colorbar min,max with norm
        cmap = mpl.cm.Greys
        norm = mpl.colors.BoundaryNorm([1,3,6,8], cmap.N)
        ax = self.df.plot(column='values', legend=True, norm = norm)
        colorbar_color = ax.get_figure().axes[1].collections[0].get_clim()
        point_color = ax.collections[0].get_clim()
        assert colorbar_color == point_color

    def test_subsets(self):
        # test min, max of subplot and plot without additional parameters
        ax = self.df.plot(column='values')
        actual_lim = ax.collections[0].get_clim()
        actual_color = ax.collections[0].get_facecolors()

        lvl1 = self.df['levels'] == '5'
        ax1 = self.df[lvl1].plot(column='values')
        actual_lim1 = ax1.collections[0].get_clim()
        actual_color1 = ax1.collections[0].get_facecolors()

        # Limits must not be equal
        assert actual_lim != actual_lim1

        # Colors of min and max for both the plot are equal(which is not correct)
        np.testing.assert_array_equal(actual_color[0],actual_color1[0])
        np.testing.assert_array_equal(actual_color[-1],actual_color1[-1])

        # test min,max for subplots and plot with norm
        # obtain min, max and actual colors from axis
        cmap = mpl.cm.Greys
        norm = mpl.colors.BoundaryNorm([1,3,6,7,9], cmap.N)
        ax = self.df.plot(column='values',  legend=True, norm = norm)
        actual_lim = ax.collections[0].get_clim()
        actual_color = ax.collections[0].get_facecolors()
        cmap = ax.collections[0].get_cmap()
        expected_color = cmap(norm(list(self.df['values'])))

        # obtain min, max and actual colors from axis of the subset
        ax1 = self.df[lvl1].plot(column='values', legend=True, norm = norm)
        actual_lim1 = ax1.collections[0].get_clim()
        actual_color1 = ax1.collections[0].get_facecolors()
        cmap1 = ax1.collections[0].get_cmap()
        expected_color1 = cmap1(norm(list(self.df.loc[lvl1]['values'])))

        # Limits must be equal
        assert actual_lim == actual_lim1

        # Colors of min and max for both the plot are equal(which is not correct)
        assert not np.array_equal(actual_color[0],actual_color1[0])
        assert not np.array_equal(actual_color[-1],actual_color1[-1])

        # Colors are equal in actual_color and expected color
        np.testing.assert_array_equal(actual_color, expected_color)
        np.testing.assert_array_equal(actual_color1, expected_color1)

        # Colors of the colors are equal
        colorbar_color = ax.get_figure().axes[1].collections[0].get_facecolor()
        colorbar_color1 = ax1.get_figure().axes[1].collections[0].get_facecolor()
        np.testing.assert_array_equal(colorbar_color, colorbar_color1)
