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
        self.points = GeoSeries(Point(i, i) for i in range(self.N))
        values = np.arange(self.N)
        levels = ['1','1','2','4','5','1','3','4','2','5']
        self.df = GeoDataFrame({'geometry': self.points, 'values': values, 'levels': levels})

        # Intializing norm
        cmap = mpl.cm.viridis_r
        self.norm = mpl.colors.BoundaryNorm([1,3,6,8], cmap.N)

    def test_min_max(self):
        # min, max from the axes must be equal to min, max of the dataframe
        # checking whether both of them are equal
        ax = self.df.plot(column='values', norm = self.norm)
        actual_lim = ax.collections[0].get_clim()
        expected_lim = (self.df['values'].min(), self.df['values'].max())
        assert actual_lim == expected_lim

        # test with passing different vmin/vmax
        ax = self.df.plot(column='values', norm = self.norm, vmin=-10, vmax=20)
        actual_colors = ax.collections[0].get_facecolors()
        assert np.any(np.equal(actual_colors[0], actual_colors[1]))

    def test_vmin_vmax(self):
        # vmin is equal to vmax, all points must have the same color
        # checking whether all the points have same color
        cmap = mpl.cm.Greys
        ax = self.df.plot(column='values', norm = self.norm, vmin=0, vmax=0)
        actual_norm = ax.collections[0].norm(list(self.df['values']))
        actual_color = cmap(actual_norm)
        np.testing.assert_array_equal(actual_color[0], actual_color[1])

    def test_norm(self):
        # norm from the axis and norm from the dataframe must be equal
        # checking whether the norm from the axis and norm from the dataframe
        ax = self.df.plot(column='values', norm = self.norm)
        actual_norm = ax.collections[0].norm(list(self.df['values']))

        expected_norm = self.norm(list(self.df['values']))
        np.testing.assert_array_equal(actual_norm,expected_norm)

    def testdf_min_max(self):
        # colorbar from the subset of the dataframe must be not_equal
        # so, 2 subset of dataframe are made
        # checking whether the colorbars have same color
        lvl1 = self.df['levels'] == '1'
        lvl2 = self.df['levels'] == '5'

        ax = self.df.plot(column='values', legend=True, norm = self.norm)
        actual_lim = ax.collections[0].get_clim()
        actual_color = ax.get_figure().axes[1].collections[0].get_facecolors()

        ax1 = self.df[lvl1].plot(column='values', legend=True, norm = self.norm)
        actual_lim1 = ax1.collections[0].get_clim()
        actual_color1 = ax1.get_figure().axes[1].collections[0].get_facecolors()

        ax2 = self.df[lvl2].plot(column='values', legend=True, norm = self.norm)
        actual_lim2 = ax2.collections[0].get_clim()
        actual_color2 = ax2.get_figure().axes[1].collections[0].get_facecolors()

        np.testing.assert_array_equal(actual_color, actual_color1)
        np.testing.assert_array_equal(actual_color, actual_color2)
        np.testing.assert_array_equal(actual_color1, actual_color2)

    def testdf_norm(self):
        # Norm must be equal(with different norm)
        # checking whether the norms are equal if different norm is initialized
        cmap = mpl.cm.tab20
        norm = mpl.colors.BoundaryNorm([0, 4, 8, 12], cmap.N)
        ax = self.df.plot(column='values', cmap = cmap, norm = norm)
        actual_norm = ax.collections[0].norm(list(self.df['values']))

        expected_norm = norm(list(self.df['values']))
        np.testing.assert_array_equal(actual_norm, expected_norm)

    def test_color(self):
        # checking whether the cmap are equal if different cmap is provided
        cmap = mpl.cm.Greys
        norm = mpl.colors.BoundaryNorm([0, 4, 8, 12], cmap.N)
        ax = self.df.plot(column='values', cmap = cmap, norm = norm)
        actual_norm = ax.collections[0].norm(list(self.df['values']))
        actual_color = cmap(actual_norm)

        expected_norm = norm(list(self.df['values']))
        expected_color = cmap(expected_norm)
        np.testing.assert_array_equal(actual_norm, expected_norm)
