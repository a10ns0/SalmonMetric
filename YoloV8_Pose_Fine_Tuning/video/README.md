Uso del script `04_inference_svo.py` (carpeta `video/`)

Propósito:
- Ejecutar inferencia de pose sobre un archivo `.svo` (ZED) o sobre un video convencional (`.mp4`, `.avi`).
- Comparar visualmente dos (o más) modelos entrenados y generar un CSV con resúmenes por frame.

Requisitos pip-only mínimos (si ya tienes un entorno con `requirements.txt`):
- ultralytics
- opencv-python
- numpy

Si quieres leer `.svo` directamente desde la cámara ZED, debes instalar el ZED SDK y los bindings Python (`pyzed`). Esto no se instala vía pip en muchos casos; sigue las instrucciones de Stereolabs: https://www.stereolabs.com/developers/

Ejemplo de uso en el equipo del laboratorio (con GPU):

```bash
# activar el entorno pip que ya tienes en la máquina del profesor
python video/04_inference_svo.py \
  --svo video/2024_07_28_15_39_06.svo \
  --models outputs/runs/salmon_pose_v1/weights/best.pt outputs/runs/salmon_pose_v124/weights/best.pt \
  --out_dir video/out --device cuda --max_frames 500 --show
```

Si no tienes `pyzed`, primero exporta `.svo` a `.mp4` con la ZED Explorer en otra máquina o pide el `.mp4` y pásalo en `--svo`.

Salida:
- `video/out/<run_name>_best.mp4` -> video anotado por cada modelo
- `video/out/combined.mp4` -> video con todos los modelos lado a lado
- `video/out/inference_summary.csv` -> resumen por frame (detecciones y keypoints)

Notas:
- Para pruebas rápidas usa `--max_frames 50`.
- Si tu entorno no tiene GPU, usa `--device cpu` (la inferencia será más lenta).
