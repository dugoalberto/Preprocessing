#!/bin/bash
set -e

PYTHON_EXEC="python3"
HOME_DIR="/mnt/home/albertodugo"
PROJECT_DIR="$HOME_DIR/Projects/Preproccessing"
SKIP_EXTRACTION=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-extraction) SKIP_EXTRACTION=true ;;
        *) echo "Argomento sconosciuto: $1"; exit 1 ;;
    esac
    shift
done

echo "Inizio del job..."
echo "Uso l'ambiente Python: $PYTHON_EXEC"
$PYTHON_EXEC --version
$PYTHON_EXEC -m pip install huggingface_hub
export HF_TOKEN=".."

$PYTHON_EXEC -m pip install --user open-clip-torch

cd "$PROJECT_DIR/segment-anything-langsplat"
$PYTHON_EXEC -m pip install --user -e .

apt-get update && apt-get install -y tar

# ── Estrazione frames ────────────────────────────────────────────────────────
TAR_PATH="$PROJECT_DIR/frames.tar"
DEST_DIR="/tmp/dataset"
FINAL_DIR="$DEST_DIR/frames"

echo "=== Inizio processo ==="
mkdir -p "$FINAL_DIR"

if [ "$SKIP_EXTRACTION" = true ]; then
    echo "Estrazione saltata (--skip-extraction attivo)."
else
    if [ -f "$TAR_PATH" ]; then
        echo "Estrazione di frames.tar in corso..."
        tar -xf "$TAR_PATH" -C "$FINAL_DIR" --strip-components=2
    else
        echo "Errore: Il file $TAR_PATH non esiste."
        exit 1
    fi
fi

# Cerca automaticamente la directory reale dei frame
FRAMES_DIR=$(find "$DEST_DIR" -type d -name "frames" | head -n 1)

if [ -z "$FRAMES_DIR" ]; then
    echo "Errore: directory frames non trovata in $DEST_DIR"
    exit 1
fi

echo "Frames trovati in: $FRAMES_DIR"

# ── Preprocessing ────────────────────────────────────────────────────────────
cd "$PROJECT_DIR/preprocessor"

echo "Avvio scannetRun.py con dataset: $FRAMES_DIR"
$PYTHON_EXEC scannetRun.py --dataset_path="$FRAMES_DIR" --resolution 256

echo "Job completato!"