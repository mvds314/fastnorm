from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="fastnorm",
    version="24.04.1",
    description="Fast evaluation of multivariate normal distribution",
    url="https://github.com/mvds314/fastnorm",
    author="Martin van der Schans",
    license="BSD",
    keywords="statistics",
    packages=["fastnorm"],
    install_requires=["numpy", "scipy", "statsmodels"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
