from setuptools import setup

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FJUtils",
    version='0.0.15',
    author="Erfan Zare Chavoshi",
    author_email="erfanzare82@yahoo.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erfanzar/",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax~=0.4.10",
        "flax~=0.7.1",
        "numpy",
        "typing~=3.7.4.3",
        "transformers~=4.31.0",
        "einops~=0.6.1",
        "optax~=0.1.7",
        "msgpack~=1.0.5",
        "tqdm==4.65.0",
        'ml_collections'
    ],
    python_requires=">=3.7, <3.11",
    license='Apache License 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
