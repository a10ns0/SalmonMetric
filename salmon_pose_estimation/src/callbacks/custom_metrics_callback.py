"""
Callbacks personalizados para YOLOv8
"""
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.metrics.evaluator import PoseEvaluator


class CustomMetricsCallback:
    """Callback para calcular métricas personalizadas durante entrenamiento"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Ruta al archivo de configuración
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.evaluator = PoseEvaluator(self.config)
        
    def on_val_end(self, validator):
        """Ejecutado al final de cada validación"""
        print("\n🔍 Calculando métricas personalizadas...")
        
        # Aquí iría la lógica de extracción de predicciones
        # Por ahora es un placeholder
        print(f"   Época {validator.epoch}: Validación completada")
    
    def on_train_end(self, trainer):
        """Ejecutado al final del entrenamiento"""
        print("\n💾 Entrenamiento completado!")
        print(f"📂 Resultados en: {trainer.save_dir}")
