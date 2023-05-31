import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nn_verify",
    version="0.1",
    author="Shubham Ugare",
    author_email="shubhamugare@gmail.com",
    description="This package provides the implementation of DNN verifier.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shubhamugare/nn_verify",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)