"""Top-level setup script for packaging the repository."""

from setuptools import setup, find_packages

setup(
    name="tree_of_thoughts",
    version="0.1.0",
    packages=find_packages(include=["tree_of_thoughts", "tasks"]),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "pandas>=1.3.0",
        "numpy>=1.19.0",
    ],
)
