"""Setup file for the package/project."""

from setuptools import find_packages, setup

setup(
    name='img2line',
    version='0.1.0',
    description='Python toolkit to fit a line to an image of a line.',
    long_description='Python toolkit to fit a line to an image of a line.',
    author='Nitish Reddy Koripalli',
    author_email='nitish.k.reddy@gmail.com',
    url='https://github.org/nitred/img2line',
    install_requires=['numpy', 'scikit-learn', 'scikit-image', 'matplotlib'],
    packages=find_packages(),
    include_package_data=True,
)
