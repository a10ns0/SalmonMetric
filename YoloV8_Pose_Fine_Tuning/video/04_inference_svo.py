#!/usr/bin/env python3
"""
Inferencia de keypoints sobre un archivo SVO (ZED) o sobre un video convencional.

Este archivo es una copia enfocada a la carpeta `video/` y preparada para
entornos pip-only (sin conda). Acepta `--device` para forzar `cpu` o `cuda`.

Uso mínimo recomendado en el lab (después de crear/activar env pip):
 python video/04_inference_svo.py \
   --svo video/2024_07_28_15_39_06.svo \
   --models outputs/runs/salmon_pose_v1/weights/best.pt outputs/runs/salmon_pose_v124/weights/best.pt \
   --out_dir video/out --device cuda --max_frames 200

Si `pyzed` no está disponible, pasa un mp4 en `--svo`.
"""
import argparse
import csv
from pathlib import Path
import sys
import time
import numpy as np
import cv2

from ultralytics import YOLO


def open_svo_or_video(path):
    try:
        import pyzed.sl as sl
    except Exception:
        return None

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(path))
    init_params.svo_real_time_mode = False
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        return None
    return zed


def safe_model_name(model_path: Path) -> str:
    # If weights are in outputs/runs/<run_name>/weights/best.pt -> use run_name
    if model_path.parent.name == 'weights' and model_path.parent.parent.name:
        return f"{model_path.parent.parent.name}_{model_path.stem}"
    return model_path.stem

#!/usr/bin/env python3
"""
Inferencia de keypoints sobre un archivo SVO (ZED) o sobre un video convencional.

Este archivo está ubicado dentro de `video/` y preparado para entornos pip-only.
Acepta `--device` para forzar `cpu` o `cuda` y `--show` para visualizar la
salida en pantalla mientras procesa frames.

Ejemplo mínimo:
 python video/04_inference_svo.py \
   --svo video/2024_07_28_15_39_06.svo \
   --models outputs/runs/salmon_pose_v1/weights/best.pt \
            outputs/runs/salmon_pose_v124/weights/best.pt \
   --out_dir video/out --device cuda --max_frames 200 --show
"""

import argparse
import csv
from pathlib import Path
import time
import numpy as np
import cv2

from ultralytics import YOLO


def open_svo_or_video(path):
    try:
        import pyzed.sl as sl
    except Exception:
        return None

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(path))
    init_params.svo_real_time_mode = False
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        return None
    return zed


def safe_model_name(model_path: Path) -> str:
    # If weights are in outputs/runs/<run_name>/weights/best.pt -> use run_name
    if model_path.parent.name == 'weights' and model_path.parent.parent.name:
        return f"{model_path.parent.parent.name}_{model_path.stem}"
    return model_path.stem


def _draw_predictions_on(frame, results, color=(0, 255, 0), kpt_radius=3, box_thickness=2):
    """Dibuja cajas y keypoints de un objeto `results[0]` sobre `frame` y devuelve frame."""
    img = frame
    try:
        r = results[0]
        # cajas
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)
                cv2.putText(img, f"{conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # keypoints
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            kpts = r.keypoints.data.cpu().numpy()
            # kpts: (n_instances, n_kpts, 3)
            for inst in kpts:
                for kp in inst:
                    x, y, v = kp
                    if v > 0.0:
                        cv2.circle(img, (int(x), int(y)), kpt_radius, color, -1)
    except Exception:
        pass
    return img


def process_with_zed(zed, models_dict, out_dir, conf_thresh=0.3, max_frames=None, show=False, overlay=False):
    import pyzed.sl as sl

    cam_info = zed.get_camera_information()
    fps = int(cam_info.camera_configuration.fps or 30)
    width = int(cam_info.camera_configuration.resolution.width)
    height = int(cam_info.camera_configuration.resolution.height)

    writers = {}
    for name in models_dict:
        writers[name] = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    combined_writer = cv2.VideoWriter(str(out_dir / "combined.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * len(models_dict), height))

    mat = sl.Mat()
    frame_idx = 0
    rows = []

    # prepare colors per model
    palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    model_colors = {name: palette[i % len(palette)] for i, name in enumerate(models_dict.keys())}

    while True:
        if max_frames and frame_idx >= max_frames:
            break
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            break
        zed.retrieve_image(mat, sl.VIEW.LEFT)
        frame = mat.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Run each model on the same frame and optionally draw overlays
        per_model_anns = {}
        for name, mdl in models_dict.items():
            results = mdl(frame, conf=conf_thresh)
            per_model_anns[name] = results

            # stats counting
            try:
                r = results[0]
                boxes_count = len(r.boxes)
                kpts_count = 0
                if hasattr(r, 'keypoints') and r.keypoints is not None:
                    kpts = r.keypoints.data.cpu().numpy()
                    for inst in kpts:
                        for kp in inst:
                            if kp[2] > 0.0:
                                kpts_count += 1
            except Exception:
                boxes_count, kpts_count = 0, 0
            rows.append((frame_idx, time.time(), name, boxes_count, kpts_count))

        # write per-model annotated videos (keep previous behavior)
        for name, results in per_model_anns.items():
            ann_img = results[0].plot()
            writers[name].write(ann_img)

        # overlay mode: draw all model predictions over the original frame
        if overlay:
            overlay_img = frame.copy()
            for name, results in per_model_anns.items():
                color = model_colors.get(name, (0, 255, 0))
                overlay_img = _draw_predictions_on(overlay_img, results, color=color)
            combined = overlay_img
        else:
            # combined image (handle single-model case)
            annotated_list = [results[0].plot() for results in per_model_anns.values()]
            if len(annotated_list) == 1:
                combined = annotated_list[0]
            else:
                combined = np.hstack(annotated_list)

        combined_writer.write(combined)

        if show:
            cv2.imshow('Inference', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    for w in writers.values():
        w.release()
    combined_writer.release()
    if show:
        cv2.destroyAllWindows()
    return rows


def process_with_cv2(video_path, models_dict, out_dir, conf_thresh=0.3, max_frames=None, show=False, overlay=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writers = {}
    for name in models_dict:
        writers[name] = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    combined_writer = cv2.VideoWriter(str(out_dir / "combined.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * len(models_dict), height))

    rows = []
    frame_idx = 0
    # prepare colors per model
    palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    model_colors = {name: palette[i % len(palette)] for i, name in enumerate(models_dict.keys())}

    while True:
        if max_frames and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        per_model_anns = {}
        for name, mdl in models_dict.items():
            results = mdl(frame, conf=conf_thresh)
            per_model_anns[name] = results

            # stats counting
            try:
                r = results[0]
                boxes_count = len(r.boxes)
                kpts_count = 0
                if hasattr(r, 'keypoints') and r.keypoints is not None:
                    kpts = r.keypoints.data.cpu().numpy()
                    for inst in kpts:
                        for kp in inst:
                            if kp[2] > 0.0:
                                kpts_count += 1
            except Exception:
                boxes_count, kpts_count = 0, 0
            rows.append((frame_idx, time.time(), name, boxes_count, kpts_count))

        # write per-model annotated videos (keep previous behavior)
        for name, results in per_model_anns.items():
            ann_img = results[0].plot()
            writers[name].write(ann_img)

        # overlay mode: draw all model predictions over the original frame
        if overlay:
            overlay_img = frame.copy()
            for name, results in per_model_anns.items():
                color = model_colors.get(name, (0, 255, 0))
                overlay_img = _draw_predictions_on(overlay_img, results, color=color)
            combined = overlay_img
        else:
            annotated_list = [results[0].plot() for results in per_model_anns.values()]
            if len(annotated_list) == 1:
                combined = annotated_list[0]
            else:
                combined = np.hstack(annotated_list)

        combined_writer.write(combined)

        if show:
            cv2.imshow('Inference', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    for w in writers.values():
        w.release()
    combined_writer.release()
    if show:
        cv2.destroyAllWindows()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo', type=str, help='Ruta al archivo SVO (ZED) o a un video (mp4, avi)')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['outputs/runs/salmon_pose_v1/weights/best.pt',
                                 'outputs/runs/salmon_pose_v124/weights/best.pt'],
                        help='Rutas a los modelos (.pt) separados por espacios')
    parser.add_argument('--out_dir', type=str, default='video/out', help='Carpeta de salida dentro de `video/`')
    parser.add_argument('--conf', type=float, default=0.3, help='Confianza mínima')
    parser.add_argument('--max_frames', type=int, default=None, help='Máximo frames a procesar (útil para prueba)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device para inferencia')
    parser.add_argument('--show', action='store_true', help='Mostrar la inferencia en pantalla durante el procesamiento')
    parser.add_argument('--overlay', action='store_true', help='Dibujar las predicciones de todos los modelos sobre el mismo frame (colores distintos)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar modelos y mover al device deseado
    models = {}
    for m in args.models:
        mp = Path(m)
        if not mp.exists():
            print(f"Modelo no encontrado: {m}")
            return
        name = safe_model_name(mp)
        print(f"Cargando modelo: {m} as {name}")
        mdl = YOLO(str(mp))
        # mover al device pedido, con fallback
        if args.device == 'cuda':
            try:
                import torch
                if torch.cuda.is_available():
                    mdl.to('cuda')
                    print(f"  -> modelo movido a cuda")
                else:
                    print("  ⚠️  CUDA no disponible, usando CPU")
                    mdl.to('cpu')
            except Exception:
                print("  ⚠️  No se pudo verificar CUDA, usando CPU")
                mdl.to('cpu')
        else:
            mdl.to('cpu')
        models[name] = mdl

    if args.svo is None:
        print("Por favor indica la ruta al archivo SVO o video con --svo")
        return

    svo_path = Path(args.svo)
    if not svo_path.exists():
        print(f"Archivo no encontrado: {svo_path}")
        return

    zed = open_svo_or_video(svo_path)
    if zed is not None:
        print("Leyendo con ZED SDK (pyzed)...")
        rows = process_with_zed(zed, models, out_dir, conf_thresh=args.conf, max_frames=args.max_frames, show=args.show, overlay=args.overlay)
        zed.close()
    else:
        print("pyzed no disponible o no se pudo abrir SVO. Usando OpenCV (si es mp4/avi)...")
        rows = process_with_cv2(svo_path, models, out_dir, conf_thresh=args.conf, max_frames=args.max_frames, show=args.show, overlay=args.overlay)

    csv_path = out_dir / 'inference_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'timestamp', 'model', 'boxes', 'keypoints'])
        for r in rows:
            writer.writerow(r)

    print(f"Inferencia completada. Salidas en: {out_dir}")


if __name__ == '__main__':
    main()
