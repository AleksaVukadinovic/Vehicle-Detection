#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTING_DIR="$SCRIPT_DIR/data/testing"
CHECKPOINT="$SCRIPT_DIR/checkpoints/best_detector.pt"

if [ ! -d "$TESTING_DIR" ]; then
    echo "Error: $TESTING_DIR not found" >&2
    exit 1
fi

shopt -s nullglob nocaseglob
images=("$TESTING_DIR"/*.jpg "$TESTING_DIR"/*.jpeg "$TESTING_DIR"/*.png)
shopt -u nullglob nocaseglob

if [ ${#images[@]} -eq 0 ]; then
    echo "No images found in $TESTING_DIR" >&2
    exit 1
fi

GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

cd "$SCRIPT_DIR"
"$SCRIPT_DIR/../venv/bin/python" detect.py --image "${images[@]}" --checkpoint "$CHECKPOINT" \
    | while IFS= read -r line; do
        if [[ "$line" =~ ^(.+):\ ([0-9]+)\ vehicle\(s\)\ detected\ -\>\ (.+)$ ]]; then
            filename="$(basename "${BASH_REMATCH[1]}")"
            count="${BASH_REMATCH[2]}"
            output="${BASH_REMATCH[3]}"
            color="$RED"
            [ "$count" -gt 0 ] && color="$GREEN"
            printf "%-20s ${color}${BOLD}%-15s${RESET} -> %s\n" "$filename" "$count vehicle(s)" "$output"
        elif [[ "$line" =~ ^\ \ box= ]]; then
            printf "${DIM}%s${RESET}\n" "$line"
        else
            echo "$line"
        fi
    done
