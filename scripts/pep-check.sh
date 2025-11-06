#!/usr/bin/env bash
set -e

# Get project root (one level above this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo ">> Checking with Black (showing diff)..."
uv run black --check --diff src

echo ">> Checking with Ruff..."
uv run ruff check src

echo ">> Done."
