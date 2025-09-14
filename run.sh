#!/bin/zsh
# Activate venv and run main.py
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate
pip install -r requirements.txt
python main.py
