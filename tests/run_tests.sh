#!/bin/bash

# Install the package using Poetry
poetry install --without dev,docs

# Source the .env file for API keys
# source .env 

# Activate the virtual environment
source "$(poetry env info --path)/bin/activate"

# Run pytest
pytest

# Deactivate the virtual environment
deactivate

# Remove the virtual environment
rm -rf "$(poetry env info --path)"