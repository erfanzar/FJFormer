from setuptools import setup

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fjformer",
    version='0.0.8',
    author="Erfan Zare Chavoshi",
    author_email="erfanzare82@yahoo.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erfanzar/",
    packages=setuptools.find_packages(),
    install_requires=[
        "chex",
        "typing",
        "jax>=0.4.10",
        "transformers>=4.34.0",
        "flax>=0.7.1",
        "optax>=0.1.7",
        "einops",
        "msgpack",
        "ml_collections",
        "einops",
        "datasets",
        "IPython"
    ],
    python_requires=">=3.7",
    license='Apache License 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)