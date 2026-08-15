#!/bin/bash
# HebrewScribe — macOS dev launcher (double-click in Finder to run)
# Activates the local venv if present, then launches the app.

cd "$(dirname "$0")"

# Try .venv-mac first, then .venv
if [ -f ".venv-mac/bin/activate" ]; then
    source .venv-mac/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Stock macOS has no bare "python" — prefer it only inside an activated venv.
if command -v python >/dev/null 2>&1; then
    python -m hebrewscribe
else
    python3 -m hebrewscribe
fi
