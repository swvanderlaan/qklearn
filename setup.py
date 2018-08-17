from setuptools import setup, find_packages

setup(
    name='qklearn',
    version='0.1',
    packages=['qklearn'],
    license='MIT',
    description='A package allowing for parallelization of sklearn-based Machine Learning on qsub based HPC architectures.',
    long_description=open('README.txt').read(),
    install_requires=['matplotlib', 'numpy', 'pandas', 'joblib'],
    url='https://github.com/tbezemer/qklearn.git',
    author='Tim Bezemer',
    author_email='tim@timbezemer.com'
)