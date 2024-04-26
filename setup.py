from setuptools import setup

setup(
    name="fastnorm",
    version="24.04",
    description="Fast evaluation of multivariate normal distribution",
    url="https://github.com/mvds314/fastnorm",
    author="Martin van der Schans",
    license="BSD",
    keywords="statistics",
    packages=["fastnorm"],
    install_requires=["numpy", "scipy", "statsmodels"],
)
