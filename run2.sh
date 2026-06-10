#!/bin/bash
set -e

PYTHON_EXEC="python3"
HOME_DIR="/mnt/home/albertodugo/"
PROJECT_DIR="$HOME_DIR/Projects/Preproccessing"

echo "Inizio del job..."
echo "Uso l'ambiente Python: $PYTHON_EXEC"
$PYTHON_EXEC --version

$PYTHON_EXEC -m pip install --user open-clip-torch

cd "$PROJECT_DIR/segment-anything-langsplat"
$PYTHON_EXEC -m pip install --user -e .

apt-get update && apt-get install -y tar

cd "$PROJECT_DIR/preprocessor"

echo "Avvio scannetRun.py con dataset: $FRAMES_DIR"
$PYTHON_EXEC scannetRun.py --resolution 256 --workers_per_gpu 10

echo "Job completato!"