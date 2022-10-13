# Installation and Setup

[pypi-badge]: https://badge.fury.io/py/invertedai.svg
[pypi-link]: https://pypi.org/project/invertedai/

## Published Package
For installing the Python package from PyPI [![PyPI][pypi-badge]][pypi-link]:

```bash
pip install invertedai
```

The `invertedai` package has minimal dependencies on other python packages. However, it is always recommended to install the package in a virtual environment. 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install invertedai
```


## Build and Install Manually
To install the latest version, clone the projects [github repository](https://github.com/inverted-ai/invertedai.git)
and install the package:
```bash
git clone https://github.com/inverted-ai/invertedai.git
cd invertedai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```
