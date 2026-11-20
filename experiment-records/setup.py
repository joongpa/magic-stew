"""Optional native search kernels; observation and search policy stay in Python."""
import os
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    'live_recognition._search_kernels',
    ['src/live_recognition/_search_kernels.c'],
    extra_compile_args=['/O2', '/fp:strict'] if os.name == 'nt' else ['-O3', '-ffp-contract=off'],
    optional=True,
)])
