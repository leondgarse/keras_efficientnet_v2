from setuptools import find_packages
from setuptools import setup

setup(
    name="keras-efficientnet-v2",
    version="1.0.0",
    author="Leondgarse",
    author_email="leondgarse@google.com",
    url="https://github.com/leondgarse/Keras_efficientnet_v2",
    description="keras efficientnet v2",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "tensorflow",
    ],
    packages=find_packages(),
    license="Apache 2.0",
)
