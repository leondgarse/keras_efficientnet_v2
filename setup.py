""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('keras_efficientnet_v2/version.py').read())
setup(
    name="keras-efficientnet-v2",
    version=__version__,
    description="(Unofficial) keras efficientnet v2",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/leondgarse/keras_efficientnet_v2",
    author="Leondgarse",
    author_email="leondgarse@google.com",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='tensorflow keras efficientnet v2 pretrained models',
    packages=find_packages(),
    include_package_data=True,
    install_requires=["tensorflow"],
    python_requires='>=3.6',
    license="Apache 2.0",
)
