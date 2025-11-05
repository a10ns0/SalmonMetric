# Contenido para: src/callbacks/custom_metrics_callback.py
"""
Callback personalizado para calcular y registrar métricas de pose (OKS, PCK)
durante el ciclo de entrenamiento de YOLO.
"""
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

from src.metrics.evaluator import PoseEvaluator

class CustomMetricsCallback:
    """
    Callback para calcular métricas de pose (OKS, PCK) y guardarlas.
    """
    def __init__(self, keypoints_config_path: str):
        """
        Args:
            keypoints_config_path: Ruta al archivo de configuración de keypoints.
        """
        with open(keypoints_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.evaluator = PoseEvaluator(config)
        self.save_dir = None

    def on_val_end(self, validator):
        """
        Se ejecuta al final de cada época de validación.
        Extrae predicciones y ground truth para calcular las métricas.
        """
        if self.save_dir is None:
            self.save_dir = validator.save_dir

        print("\n🔍 Calculando métricas personalizadas (OKS, PCK)...")
        
        # Extraer predicciones y ground truth del validador
        preds = validator.predictions
        gt_keypoints = np.array([p['keypoints'] for p in preds])
        pred_keypoints = np.array([p['pred_keypoints'] for p in preds])
        bboxes = np.array([p['bbox'] for p in preds])
        
        # Asegurarse de que los datos tengan la forma correcta
        if gt_keypoints.ndim == 2:
            gt_keypoints = gt_keypoints.reshape(gt_keypoints.shape[0], -1, 3)
            # Aquí asumimos que la tercera columna es la visibilidad.
            # Dependiendo del formato exacto, podrías necesitar ajustar esto.
        
        # Evaluar el batch completo
        batch_metrics = self.evaluator.evaluate_batch(pred_keypoints, gt_keypoints, bboxes)
        
        # Añadir métricas al log de YOLO para que aparezcan en los resultados
        for key, value in batch_metrics.items():
            validator.metrics.results_dict[key] = value
            
        print("   ✅ Métricas personalizadas calculadas y añadidas.")

    def on_train_end(self, trainer):
        """Se ejecuta al final de todo el entrenamiento."""
        if self.save_dir:
            final_csv_path = self.save_dir / 'custom_metrics_summary.csv'
            self.evaluator.save_results_to_csv(final_csv_path)
            print(f"\n💾 Resumen de métricas personalizadas guardado en: {final_csv_path}")

    def get_callbacks(self):
        """Devuelve un diccionario de los eventos que YOLO debe manejar."""
        return {
            'on_val_end': self.on_val_end,
            'on_train_end': self.on_train_end
        }
