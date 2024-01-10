from setuptools import setup

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FJFormer",
    version="0.0.25",
    author="Erfan Zare Chavoshi",
    author_email="erfanzare82@yahoo.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erfanzar/FJFormer",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax>=0.4.16",
        "jaxlib>=0.4.16",
        "numpy~=1.26.2",
        "typing~=3.7.4.3",
        "flax~=0.7.5",
        "chex~=0.1.84",
        "ipython~=8.17.2",
        "datasets~=2.14.7",
        "einops~=0.6.1",
        "msgpack~=1.0.7",
        "tqdm~=4.64.1",
        "optax~=0.1.7",
        "setuptools~=68.1.2",
        "ml_collections==0.1.1"
    ],
    python_requires=">=3.8",
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
