# Contenido para: src/utils/download_utils.py
"""
Utilidades para verificar y descargar modelos pre-entrenados de YOLO.
Este módulo es agnóstico al proyecto y puede ser reutilizado.
"""
import sys
from pathlib import Path
import requests
import warnings

# Asegurar que ultralytics esté disponible
try:
    from ultralytics import YOLO
    from ultralytics.utils.downloads import safe_download
except ImportError:
    warnings.warn("La librería 'ultralytics' no está instalada. Algunas funciones pueden no estar disponibles.")
    YOLO, safe_download = None, None

class ModelDownloader:
    """
    Gestor para verificar y descargar modelos de YOLO, ahora incluyendo YOLOv11.
    """
    
    # Directorio central para los pesos de los modelos, siguiendo el estándar de ultralytics
    MODELS_DIR = Path.home() / '.cache/torch/ultralytics/'

    # Diccionario de modelos conocidos con sus propiedades
    AVAILABLE_MODELS = {
        # --- Modelos YOLOv11 (Recomendados para este proyecto) ---
        'yolov11s-pose.pt': {
            'size_mb': 22.5, 'params_m': 11.2, 'description': 'YOLOv11 Small - Pose Estimation (Rápido y Eficiente)', 'recommended': True
        },
        'yolov11m-pose.pt': {
            'size_mb': 49.7, 'params_m': 25.9, 'description': 'YOLOv11 Medium - Pose Estimation (Balanceado)', 'recommended': False
        },
        'yolov11l-pose.pt': {
            'size_mb': 99.2, 'params_m': 52.2, 'description': 'YOLOv11 Large - Pose Estimation (Máxima Precisión)', 'recommended': False
        },

        # --- Modelos YOLOv8 (Para referencia, no recomendados aquí) ---
        'yolov8s-pose.pt': {
            'size_mb': 22.9, 'params_m': 11.2, 'description': 'YOLOv8 Small - Pose Estimation (Baseline)', 'recommended': False
        },
        'yolov8m-pose.pt': {
            'size_mb': 52.2, 'params_m': 25.9, 'description': 'YOLOv8 Medium - Pose Estimation', 'recommended': False
        },
        'yolov8l-pose.pt': {
            'size_mb': 87.4, 'params_m': 43.7, 'description': 'YOLOv8 Large - Pose Estimation', 'recommended': False
        },
    }

    @staticmethod
    def get_model_path(model_name: str) -> Path:
        """Devuelve la ruta esperada para un modelo en el caché local."""
        return ModelDownloader.MODELS_DIR / model_name

    @staticmethod
    def check_model_exists(model_name: str) -> bool:
        """Verifica si un modelo ya existe localmente."""
        return ModelDownloader.get_model_path(model_name).exists()

    @staticmethod
    def download_model(model_name: str, verbose: bool = False):
        """
        Descarga un modelo usando la función segura de ultralytics.
        Esta función maneja la lógica de descarga, descompresión y caché.
        """
        if YOLO is None or safe_download is None:
            raise ImportError("La librería 'ultralytics' es necesaria para descargar modelos.")
            
        if model_name not in ModelDownloader.AVAILABLE_MODELS:
            warnings.warn(f"'{model_name}' no es un modelo conocido. Se intentará la descarga directa.")
        
        model_path = ModelDownloader.get_model_path(model_name)
        
        if ModelDownloader.check_model_exists(model_name):
            if verbose:
                print(f"   ✅ El modelo '{model_name}' ya existe en: {model_path}")
            return str(model_path)
            
        if verbose:
            info = ModelDownloader.AVAILABLE_MODELS.get(model_name, {})
            size = info.get('size_mb', 'desconocido')
            print(f"   📥 Descargando '{model_name}' (Tamaño: {size} MB)...")
        
        try:
            # ultralytics maneja la URL de descarga automáticamente por el nombre del archivo
            safe_download(model_name, dir=ModelDownloader.MODELS_DIR)
            
            if verbose:
                print(f"   ✅ Descarga completada. Modelo guardado en: {model_path}")
            
            return str(model_path)
        except Exception as e:
            print(f"   ❌ Fallo en la descarga del modelo '{model_name}'. Error: {e}")
            raise

