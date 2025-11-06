#!/usr/bin/env bash
set -e

# Get project root (one level above this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo ">> Formatting with Black..."
uv run black src

echo ">> Fixing with Ruff..."
uv run ruff check src --fix

echo ">> Done."
