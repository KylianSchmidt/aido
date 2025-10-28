from setuptools import find_packages, setup

requirements: list[str] = []

with open("requirements.txt", "r") as file:
    requirements.extend(file.readlines())

with open("../requirements.txt", "r") as file:
    requirements.extend(file.readlines())

setup(
    name="aido",
    packages=find_packages(),
    install_requires=requirements,
)
