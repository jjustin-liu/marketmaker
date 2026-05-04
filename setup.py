"""Build shim for the pybind11 C++ extension.

Pure metadata stays in pyproject.toml. This file exists only because
setuptools' TOML support cannot declare ext_modules.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "src.lob._match_engine",
        ["src/lob/match_engine.cpp"],
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
