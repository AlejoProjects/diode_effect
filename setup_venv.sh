#!/usr/bin/env sh
set -eu

PROJECT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_COMMAND=${PYTHON_COMMAND:-python3}
ENVIRONMENT_DIRECTORY=${ENVIRONMENT_DIRECTORY:-.venv}
ENVIRONMENT_PATH="$PROJECT_DIRECTORY/$ENVIRONMENT_DIRECTORY"

echo "Creating virtual environment at $ENVIRONMENT_PATH"
"$PYTHON_COMMAND" -m venv "$ENVIRONMENT_PATH"

echo "Updating packaging tools"
"$ENVIRONMENT_PATH/bin/python" -m pip install --upgrade pip setuptools wheel

echo "Installing project dependencies"
"$ENVIRONMENT_PATH/bin/python" -m pip install --requirement "$PROJECT_DIRECTORY/requirements.txt"

echo "Registering the Jupyter kernel"
"$ENVIRONMENT_PATH/bin/python" -m ipykernel install --user \
  --name diode-effect --display-name "Python (diode-effect)"

echo "Environment ready. Activate it with:"
echo ". $ENVIRONMENT_PATH/bin/activate"
echo "Then start JupyterLab with: python -m jupyterlab"
