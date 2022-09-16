# Get Started

Inverted AI has trained cutting-edge realistic behavioral driving models that are human-like and close the SIM2Real. Our API provides access to these behavioral models and can be used for many tasks in autonomous vehicle (AV) industry.

In this quickstart tutorial, you’ll run a simple sample AV simulation with Inverted AI Python API. Along the way, you’ll learn key concepts and techniques that are fundamental to using the API for other tasks. In particular, you will be familiar with two main Inverted AI models:

- Drive
- Initialize

## Installation

[pypi-badge]: https://badge.fury.io/py/invertedai_drive.svg
[pypi-link]: https://pypi.org/project/invertedai-drive/

To install use [![PyPI][pypi-badge]][pypi-link]:

```bash
pip install invertedai
```

## Setting up

Import the _invertedai_ package and set the API key with **add_apikey** method.

Refer to the [product page](https://www.inverted.ai) to get (or recharge) your API key.

```python

import invertedai as iai
iai.add_apikey("XXXXXXXXXXXXXX")
```

## Initialize

## Drive

## Running demo locally

Download the examples [directory](https://github.com/inverted-ai/invertedai-drive/blob/master/examples) and run:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/jupyter notebook Drive-Demo.ipynb
```

## Running demo in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/Colab-Demo.ipynb)
