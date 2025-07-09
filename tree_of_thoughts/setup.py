"""Packaging setup for installing the ``tree_of_thoughts`` package."""

from setuptools import setup, find_packages

setup(
    name="tree_of_thoughts",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "pandas>=1.3.0",
        "numpy>=1.19.0",
    ],
)
