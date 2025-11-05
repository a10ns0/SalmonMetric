"""
Evaluador principal que integra todas las métricas
"""
import numpy as np
import pandas as pd
from typing import Dict
from .pck import PCKMetric
from .oks import OKSMetric


class PoseEvaluator:
    """Evaluador completo de métricas para pose estimation"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Diccionario con configuración de keypoints
        """
        self.keypoint_names = config['keypoints']['names']
        self.num_keypoints = config['keypoints']['num_keypoints']
        
        self.pck_metrics = {
            f'pck@{th}': PCKMetric(threshold=th)
            for th in config['validation']['custom_metrics']['pck_thresholds']
        }
        
        self.oks_metric = OKSMetric(
            sigmas=np.array(config['keypoints']['oks_sigmas'])
        )
        
        self.results = []
        
    def evaluate_batch(
        self,
        predictions: Dict,
        ground_truth: Dict
    ) -> Dict[str, float]:
        """
        Evalúa un batch de predicciones
        """
        metrics = {}
        
        for name, pck_metric in self.pck_metrics.items():
            pck_global, pck_per_kpt = pck_metric.compute(
                predictions['keypoints'],
                ground_truth['keypoints'],
                predictions['bboxes']
            )
            metrics[name] = pck_global
            
            for i, kpt_name in enumerate(self.keypoint_names):
                metrics[f'{name}_{kpt_name}'] = pck_per_kpt[i]
        
        oks_scores = self.oks_metric.compute_batch(
            predictions['keypoints'],
            ground_truth['keypoints'],
            predictions['bboxes'],
            ground_truth['visibilities']
        )
        metrics['oks_mean'] = oks_scores.mean()
        metrics['oks_std'] = oks_scores.std()
        
        valid_detections = (oks_scores > 0.5).sum()
        total_predictions = len(predictions['keypoints'])
        total_ground_truth = len(ground_truth['keypoints'])
        
        metrics['precision'] = valid_detections / total_predictions if total_predictions > 0 else 0
        metrics['recall'] = valid_detections / total_ground_truth if total_ground_truth > 0 else 0
        
        return metrics
    
    def add_result(self, epoch: int, metrics: Dict):
        """Agregar resultado de una época"""
        metrics['epoch'] = epoch
        self.results.append(metrics)
    
    def get_summary(self) -> pd.DataFrame:
        """Obtener resumen de todas las métricas"""
        return pd.DataFrame(self.results)
    
    def save(self, filepath: str):
        """Guardar resultados a CSV"""
        df = self.get_summary()
        df.to_csv(filepath, index=False)
        print(f"✅ Métricas guardadas en: {filepath}")
