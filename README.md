<img src="docs/_static/logos/light_logo.png" width="40%" style="margin:auto; display:block;">

Framework for translating detector design into an optimization problem. By interpolating the results of simulations with slightly different geometries, it can estimate the best set of detector parameters with the help of gradient descent.

This framework fragments the workflow into b2luigi Tasks for parallel simulations and training of ML models on GPUs.

## Installation

Installation with 'setup.py' requires pip>=24.0 and python>=3.12:
1. Git clone this repository (or download the code)
2. `cd` to the directory
3. Install the `aido` package with

```
pip install .
```

The use of a virtual environment is highly recommended.

## Documentation

[aido.readthedocs.io](https://aido.readthedocs.io/en/latest/index.html)

## Publications

 - [arxiv:2502.02152](https://arxiv.org/abs/2502.02152): Describes the methodology and introduces an application to sampling calorimeters.
 - [Poster at ACAT 2025](https://indico.cern.ch/event/1488410/contributions/6562879/attachments/3129592/5551767/aido_poster_20250901_print_ready.pdf)
