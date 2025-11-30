#!/usr/bin/env bash
# Installer minimal para el script de video (pip-only)
# Ejecutar en el entorno Python del profesor (no hace conda).

set -euo pipefail

REQ_FILE="$(dirname "$0")/requirements_video.txt"
if [ ! -f "$REQ_FILE" ]; then
  echo "No encontré $REQ_FILE"
  exit 1
fi

echo "Instalando dependencias mínimas desde $REQ_FILE..."
python -m pip install --upgrade pip
python -m pip install -r "$REQ_FILE"

echo "Instalación completada. Puedes ejecutar:"
echo "python video/04_inference_svo.py --svo video/2024_07_28_15_39_06.svo --models outputs/runs/salmon_pose_v1/weights/best.pt outputs/runs/salmon_pose_v124/weights/best.pt --out_dir video/out --device cuda --max_frames 200 --show --overlay"
