from setuptools import setup, find_packages

setup(
    name='spaotsc',
    version='0.2',
    packages=find_packages(exclude=['tests*']),
    description='Spatial optimal transport for single-cell transcriptomics data.',
    author='Zixuan Cang',
    author_email='cangzx@gmail.com'
)
