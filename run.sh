#!/usr/bin/env bash
# Open the live dashboard in your browser. Runs install.sh first if needed.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run - setting up (this happens only once) ..."
  bash install.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m embodied_agent.gui "$@"
