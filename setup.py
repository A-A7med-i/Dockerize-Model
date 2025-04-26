from setuptools import find_namespace_packages, setup

PROJECT_NAME = "Dockerize Model"
VERSION = "0.0.0"
AUTHOR = "Ahmed"
REPO_NAME = "Fruits-Classification"

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description="A small package for simple cnn app",
    packages=find_namespace_packages(include=["src.*", "template.*"]),
)
