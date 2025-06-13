from setuptools import setup, Extension

module = Extension(
    name='mykmeanspp',                   # This is the import name in Python
    sources=['kmeansmodule.c'],          # Your C source file
)

setup(
    name='mykmeanspp',
    version='1.0',
    description='K-means++ C extension for clustering',
    ext_modules=[module],
)
