from setuptools import find_packages, setup

setup(
    name='estallidos',
    packages=find_packages(include=['estallidos']),
    version='0.1.0',
    description='Librería que condensa funciones útiles para investigación Fondecyt-Estallidos',
    author='',
    license='CC',
    install_requires=['pandas', 'pandas-gbq', 'nltk'],
)
