from setuptools import Extension, setup

module = Extension("mykmeanspp", sources=['kmeansmodule.c'])
setup(name="mykmeanspp",version='1.0',description="Python wrapper for C fit implementation", ext_modules=[module])