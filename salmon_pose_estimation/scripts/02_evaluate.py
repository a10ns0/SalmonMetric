"""
Script de evaluación con métricas personalizadas
"""
import yaml
from pathlib import Path
from ultralytics import YOLO
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.evaluator import PoseEvaluator


def main():
    print("=" * 80)
    print("📊 EVALUACIÓN DE MODELO YOLOv8-POSE")
    print("=" * 80)
    
    # Cargar modelo
    model_path = 'outputs/runs/salmon_pose_v1/weights/best.pt'
    print(f"\n🔍 Cargando modelo: {model_path}")
    model = YOLO(model_path)
    
    # Cargar configuración
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('config/keypoints_config.yaml', 'r') as f:
        kpt_config = yaml.safe_load(f)
    
    # Validar
    print(f"\n⏳ Ejecutando validación...")
    metrics = model.val(
        data=config['paths']['data_yaml'],
        split='test',
        batch=16,
        imgsz=config['model']['input_size'],
        conf=config['validation']['conf_threshold'],
        iou=config['validation']['iou_threshold'],
        save_json=True,
        plots=True
    )
    
    # Imprimir métricas automáticas
    print("\n" + "=" * 80)
    print("📈 MÉTRICAS AUTOMÁTICAS (YOLOv8)")
    print("=" * 80)
    
    if hasattr(metrics, 'box'):
        print(f"\nBounding Box:")
        print(f"  mAP@0.5:   {metrics.box.map50:.3f}")
        print(f"  mAP@0.75:  {metrics.box.map75:.3f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
    
    if hasattr(metrics, 'pose'):
        print(f"\nPose (Keypoints):")
        print(f"  mAP@0.5:   {metrics.pose.map50:.3f}")
        print(f"  mAP@0.75:  {metrics.pose.map75:.3f}")
        print(f"  mAP@0.5:0.95: {metrics.pose.map:.3f}")
    
    if hasattr(metrics, 'box'):
        print(f"\nGeneral:")
        print(f"  Precision: {metrics.box.p:.3f}")
        print(f"  Recall:    {metrics.box.r:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ EVALUACIÓN COMPLETADA")
    print("=" * 80)


if __name__ == '__main__':
    main()
