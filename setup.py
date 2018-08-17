from setuptools import setup, find_packages

setup(
    name='qklearn',
    version='0.1',
    packages=find_packages(),
    license='MIT',
    description='A package allowing for parallelization of sklearn-based Machine Learning on qsub based HPC architectures.',
    long_description=open('README.txt').read(),
    install_requires=['numpy', 'pandas', 'joblib'],
    url='https://github.com/tbezemer/qklearn.git',
    author='Tim Bezemer',
    author_email='tim@timbezemer.com'
)