#!/usr/bin/env bash
set -euo pipefail

# Placeholder bootstrap script; fill in once Shadeform instance access is verified.
# Expected to install system dependencies and Python packages.

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
