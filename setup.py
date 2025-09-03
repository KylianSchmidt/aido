from setuptools import find_packages, setup

with open("requirements.txt", "r") as file:
    requirements = file.readlines()

setup(
    name="aido",
    version="0.1.3",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.12.3",
    license="",
    author="Kylian Schmidt, Dr. Jan Kieseler",
    description="Tool for the optimization of detectors with machine learning",
)
