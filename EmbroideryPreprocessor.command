#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Load API key from .env file if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

python launcher.py
