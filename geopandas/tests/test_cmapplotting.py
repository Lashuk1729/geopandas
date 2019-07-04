from __future__ import absolute_import, division

import itertools
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from shapely.geometry import Point, Polygon, LineString

from geopandas import GeoSeries, GeoDataFrame, plotting
from geopandas.tests.test_plotting import _check_colors

import pytest

@pytest.fixture(autouse=True)
def close_figures(request):
    yield
    plt.close('all')

try:
    cycle = mpl.rcParams['axes.prop_cycle'].by_key()
    MPL_DFT_COLOR = cycle['color'][0]
except KeyError:
    MPL_DFT_COLOR = matplotlib.rcParams['axes.color_cycle'][0]

class TestPointPlotting:
    def setup_method(self):
        # Intializing the dataframe
        self.N = 10
        self.points = GeoSeries(Point(i, i, i) for i in range(self.N))

        values = np.arange(self.N)
        levels = ['1','1','1','1','1','1','2','2','2','2']

        self.df = GeoDataFrame({'geometry': self.points, 'values': values, 'levels': levels})

    def test_customnorm(self):
        # test color without additional parameters
        ax = self.df.plot(column='values')
        actual_color = ax.collections[0].get_facecolors()
        cmap = ax.collections[0].get_cmap()
        expected_color = cmap(np.arange(self.N) / (self.N - 1))
        np.testing.assert_array_equal(actual_color, expected_color)

        # test color with providing norm
        norm = mpl.colors.Normalize(vmin = 0, vmax = 12)
        ax = self.df.plot(column='values', norm = norm)
        actual_color = ax.collections[0].get_facecolors()
        cmap = ax.collections[0].get_cmap()
        expected_color = cmap(norm(list(self.df['values'])))
        np.testing.assert_array_equal(actual_color, expected_color)

        # test color with providing BoundaryNorm
        cmap = mpl.cm.Greys
        norm = mpl.colors.Normalize(vmin = 0, vmax = 12)
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
        norm = mpl.colors.Normalize(vmin = 0, vmax = 12)
        ax = self.df.plot(column='values', legend=True, norm = norm)
        colorbar_color = ax.get_figure().axes[1].collections[0].get_clim()
        point_color = ax.collections[0].get_clim()
        assert colorbar_color == point_color

        # test color with providing BoundaryNorm
        cmap = mpl.cm.Greys
        norm = mpl.colors.BoundaryNorm([1,3,6,8], cmap.N)
        ax = self.df.plot(column='values', legend=True, norm = norm)
        colorbar_color = ax.get_figure().axes[1].collections[0].get_clim()
        point_color = ax.collections[0].get_clim()
        assert colorbar_color == point_color

class TestCmapPolygonPlotting:
    def setup_method(self):
        # Intializing the dataframe
        self.N = 10
        self.points = GeoSeries(Point(i, i, i) for i in range(self.N)).buffer(0.5)

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

        # test color with providing norm
        norm = mpl.colors.Normalize(vmin = 0, vmax = 12)
        ax = self.df.plot(column='values', norm = norm)
        actual_color = ax.collections[0].get_facecolors()
        cmap = ax.collections[0].get_cmap()
        expected_color = cmap(norm(list(self.df['values'])))
        np.testing.assert_array_equal(actual_color, expected_color)

        # test color with providing BoundaryNorm
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
        norm = mpl.colors.Normalize(vmin = 0, vmax = 12)
        ax = self.df.plot(column='values', legend=True, norm = norm)
        colorbar_color = ax.get_figure().axes[1].collections[0].get_clim()
        point_color = ax.collections[0].get_clim()
        assert colorbar_color == point_color

        # test color with providing BoundaryNorm
        cmap = mpl.cm.Greys
        norm = mpl.colors.BoundaryNorm([1,3,6,8], cmap.N)
        ax = self.df.plot(column='values', legend=True, norm = norm)
        colorbar_color = ax.get_figure().axes[1].collections[0].get_clim()
        point_color = ax.collections[0].get_clim()
        assert colorbar_color == point_color

    def test_subsets_nonorm(self):
        # test min, max of subplot without additional parameters
        ax = self.df.plot(column='values')
        actual_lim = ax.collections[0].get_clim()

        lvl1 = self.df['levels'] == '5'
        ax1 = self.df[lvl1].plot(column='values')
        actual_lim1 = ax1.collections[0].get_clim()

        # Limits must not be equal
        assert actual_lim != actual_lim1

    def test_subsets_norm(self):
        # test min,max for subplots and plot with norm
        lvl1 = self.df['levels'] == '5'

        cmap = mpl.cm.Greys
        norm = mpl.colors.BoundaryNorm([1,3,6,7,9], cmap.N)
        ax = self.df.plot(column='values',  legend=True, norm = norm)
        actual_lim = ax.collections[0].get_clim()

        # obtain min, max and actual colors from axis of the subset
        ax1 = self.df[lvl1].plot(column='values', legend=True, norm = norm)
        actual_lim1 = ax1.collections[0].get_clim()

        # Limits must not be equal
        assert actual_lim == actual_lim1

    def test_subsets_color(self):
        # test min,max for subplots and plot with norm
        # obtain min, max and actual colors from axis
        lvl1 = self.df['levels'] == '5'
        cmap = mpl.cm.Greys
        norm = mpl.colors.BoundaryNorm([1,3,6,7,9], cmap.N)
        ax = self.df.plot(column='values',  legend=True, norm = norm)
        actual_color = ax.collections[0].get_facecolors()
        cmap = ax.collections[0].get_cmap()
        expected_color = cmap(norm(list(self.df['values'])))

        # obtain min, max and actual colors from axis of the subset
        ax1 = self.df[lvl1].plot(column='values', legend=True, norm = norm)
        actual_color1 = ax1.collections[0].get_facecolors()
        cmap1 = ax1.collections[0].get_cmap()
        expected_color1 = cmap1(norm(list(self.df.loc[lvl1]['values'])))

        # Colors of min and max for both the plot are not equal
        assert not np.array_equal(actual_color[0],actual_color1[0])
        assert not np.array_equal(actual_color[-1],actual_color1[-1])

class TestCMapLinestringPlotting:
    def setup_method(self):
        self.N = 10
        self.values = np.arange(self.N)
        levels = ['1','1','1','1','1','1','2','2','2','2']
        self.lines = GeoSeries([LineString([(0, i), (4, i + 0.5), (9, i)]) for i in range(self.N)])
        self.df = GeoDataFrame({'geometry': self.lines, 'values': self.values, 'levels': levels})

    def test_customnorm(self):
        from geopandas.plotting import plot_linestring_collection
        fig, ax = plt.subplots()
        coll = plot_linestring_collection(ax, self.lines, self.values)
        fig.canvas.draw_idle()
        cmap = plt.get_cmap()
        expected_colors = cmap(np.arange(self.N) / (self.N - 1))
        np.testing.assert_array_equal(coll.get_color(), expected_colors)
        ax.cla()

        fig, ax = plt.subplots()
        cmap = mpl.cm.Greys
        norm = mpl.colors.BoundaryNorm([1,3,6,8], cmap.N)
        coll = plot_linestring_collection(ax, self.lines, self.values, norm = norm)
        fig.canvas.draw_idle()

        cmap = plt.get_cmap()
        expected_colors = cmap(norm(list(self.df['values'])))
        np.testing.assert_array_equal(coll.get_color(), expected_colors)
        ax.cla()

    def test_subset_customnorm(self):
        from geopandas.plotting import plot_linestring_collection
        fig, ax = plt.subplots()

        lvl1 = self.df['levels'] == '1'
        cmap = mpl.cm.Greys
        norm = mpl.colors.BoundaryNorm([1,2,4], cmap.N)
        coll = plot_linestring_collection(ax, self.lines[0:3], self.values[0:3], norm = norm)
        fig.canvas.draw_idle()

        cmap = plt.get_cmap()
        expected_color = cmap(np.arange(self.N) / (self.N - 1))

        np.testing.assert_array_equal(coll.get_colors()[0], expected_color[0])
        np.testing.assert_array_equal(coll.get_colors()[-1], expected_color[-1])

        ax.cla()
