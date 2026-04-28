#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/"
INCLUDE_FILE="${SCRIPT_DIR}/sync_to_olivia.include"
DST="olivia:finetunes/"

if [ ! -f "$INCLUDE_FILE" ]; then
  echo "Missing include file: $INCLUDE_FILE" >&2
  exit 1
fi

rsync -av --files-from="$INCLUDE_FILE" "$SRC" "$DST"
