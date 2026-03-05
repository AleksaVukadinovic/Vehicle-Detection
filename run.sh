#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment …"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies …"
pip install -q -r requirements.txt

MODE="${1:-train}"

case "$MODE" in
    train)
        shift 2>/dev/null || true
        echo "Starting training (dataset will be downloaded on first run) …"
        python train.py "$@"
        ;;
    detect)
        shift
        echo "Running inference …"
        python inference.py "$@"
        ;;
    *)
        echo "Usage:"
        echo "  ./run.sh train  [--epochs N] [--batch_size N] [--lr F]"
        echo "  ./run.sh detect path/to/image.jpg [--output out.png] [--conf 0.3]"
        exit 1
        ;;
esac
