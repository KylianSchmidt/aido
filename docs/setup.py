from setuptools import find_packages, setup

with open("requirements.txt", "r") as file:
    requirements = file.readlines()

setup(
    name="aido",
    packages=find_packages(),
    install_requires=requirements,
)
