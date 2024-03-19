from setuptools import setup, Extension

module = Extension ('add', sources=['add.pyx'])

setup(
    name='cythonTest',
    version='1.0',
    author='jetbrains',
    ext_modules=[module]
)
