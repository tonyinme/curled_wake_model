from setuptools import setup, find_packages
import os

setup(
    name="curled_wake_model",
    version="0.1.0",
    packages=find_packages(
        where=".",
        include=["curled_wake_model*"]
    ),
    install_requires=[
        line.strip() for line in open("requirements.txt") if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.12",
    include_package_data=True,
)
