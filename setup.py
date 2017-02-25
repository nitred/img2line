"""Setup file for the package/project."""

from setuptools import find_packages, setup

setup(
    name='img2line',
    version='0.0.1',
    description='Python toolkit to fit a line to an image of a line.',
    long_description='Python toolkit to fit a line to an image of a line.',
    author='Nitish Reddy Koripalli',
    author_email='nitish.k.reddy@recogizer.de',
    url='https://github.org/nitred/img2line',
    install_requires=[],
    packages=find_packages(),
    include_package_data=True,
)
