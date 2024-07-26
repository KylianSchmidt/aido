from setuptools import setup, find_packages

with open("requirements.txt", "r") as file:
    requirements = file.readlines()

setup(
    name="aido",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
    license="",
    author="Kylian Schmidt, Dr. Jan Kieseler",
    description="Tool for the optimization of detectors with machine learning"
)
