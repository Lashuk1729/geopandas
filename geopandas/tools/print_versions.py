'''
Utility Methods to print system infomation

adapted from :func:'sklearn.show_versions'
'''

import importlib
import locale
import os
import platform
import struct
import subprocess
import sys

def _get_sys_info():
    '''
    System Information

    Return:
    sys_info as a dict
        system and Python Version information
    '''
    python = sys.version.replace('\n', ' ')
    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
        ("processor",platform.processor()),
    ]
    return dict(blob)

def _get_deps_info():
    '''
    Overview of the installed module version of main dependencies

    Return:
    deps_info: dict
        installed module version information on Python libraries
    '''
    deps = [
        "pandas",
        "pytest",
        "pip",
        "setuptools",
        "Cython",
        "numpy",
        "conda-forge",
        "shapely",
        "fiona",
        "pyproj",
        "six",
        "rtree",
    ]

    def get_version(module):
        return module.__version__

    deps_info = {}

    for mod_name in deps:
        try:
            if mod_name in sys.modules:
                mod = sys.modules[mod_name]
            else:
                mod = importlib.import_module(mod_name)
            ver = get_version(mod)
            deps_info[mod_name] = ver
        except ImportError:
            deps_info[mod_name] = None

    return deps_info

def show_versions():
    '''
    Print debugging information and installed module versions.
    '''
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print('\nSystem:')
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k = k, stat = stat))

    print('\nPython deps:')
    for k, stat in deps_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))
