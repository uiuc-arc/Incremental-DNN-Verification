import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the content of the requirements.txt file	
with open('requirements.txt', 'r', encoding='utf-8') as f:	
    requirements = f.read().splitlines()

setuptools.setup(
    name="nn_verify",
    version="0.1",
    author="Shubham Ugare",
    author_email="shubhamugare@gmail.com",
    description="This package provides the implementation of DNN verifier.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shubhamugare/nn_verify",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
