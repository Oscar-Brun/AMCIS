"""Build Cython extensions for SOLEDGE adjoint MC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent

extensions = [
    Extension(
        "adjoint_mc.core._tracker",
        ["src/adjoint_mc/core/_tracker.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        annotate=False,
    ),
)
