from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension
import os

# Library information
LIB_NAME = "ih"
VERSION = "0.1.0"

# C++ extension
ext_modules = [
    Pybind11Extension(
        f"{LIB_NAME}._core",
        ["src/entropy_core.cpp", "src/entropy_module.cpp", "src/rule_finder.cpp"],
        include_dirs=["include"],
        cxx_std=17,
    ),
]

setup(
    name="ih-lib",
    version=VERSION,
    author="Your Name",
    description="Information-Entropy analysis library",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    packages=[LIB_NAME],
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19",
        "pybind11>=2.6",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)