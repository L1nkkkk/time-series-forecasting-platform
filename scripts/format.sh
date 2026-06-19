#!/usr/bin/env sh
set -eu

ruff check --fix .
ruff format .
