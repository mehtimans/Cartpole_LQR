from setuptools import find_packages
from distutils.core import setup

setup(
    name='CartPoleLQR',
    version='1.0.0',
    author='mahdi mansouri',
    packages=find_packages(),
    author_email='mmhdimansouri@gmail.com',
    description='LQR controller for cartpole',
    install_requires=['isaacgym',
                      'matplotlib',
                      'tqdm',
                      'numpy==1.23.5',]
)
