import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import Point, MultiPoint
from geopandas import GeoSeries, GeoDataFrame, plotting
try:
    if sys.version_info < (2, 7):
        import unittest2
    else:
        raise ImportError()
except ImportError:
    import unittest

class TestPlotting(unittest.TestCase):
    def setUp(self):
        self.N = 10
        self.points = GeoSeries(Point(i, i) for i in range(self.N))
        values = np.arange(self.N)*10
        self.df = GeoDataFrame({'geometry': self.points, 'values': values})
        bounds = np.array([-10, 0, 10, 30, 50, 70, 90])
        self.norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    def test_boundmin(self):
        bounds = np.array([-10, 0, 10, 30, 50, 70, 90])
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        self.assertEqual(min(self.norm.boundaries), min(norm.boundaries),'Test ran successfully, both min are equal')

    def test_boundmax(self):
        bounds = np.array([-10, 0, 10, 30, 50, 70, 90])
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        self.assertEqual(max(self.norm.boundaries), max(norm.boundaries),'Test ran successfully, both max are equal')

    def test_dfmax(self):
        N = 10
        points = GeoSeries(Point(i, i) for i in range(self.N))
        values = np.arange(self.N)*10
        df = GeoDataFrame({'geometry': points, 'values': values})
        self.assertEqual(df['values'].max(), self.df['values'].max())

    def test_dfmin(self):
        N = 10
        points = GeoSeries(Point(i, i) for i in range(self.N))
        values = np.arange(self.N)*10
        df = GeoDataFrame({'geometry': points, 'values': values})
        self.assertEqual(df['values'].min(), self.df['values'].min())

if __name__ == '__main__':
    unittest.main()
