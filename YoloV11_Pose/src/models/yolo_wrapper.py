# Contenido para: src/models/yolo_wrapper.py
"""
Wrapper para YOLOv11-Pose, adaptado para la experimentación.
"""
import sys
from pathlib import Path
import yaml

# Agregar src al path para asegurar importaciones correctas
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
from src.utils.download_utils import ModelDownloader

class YOLOv11PoseTrainer:
    """Trainer modular para YOLOv11-Pose con configuración externa."""

    def __init__(self, training_config_path: str):
        """
        Carga la configuración de entrenamiento desde un archivo YAML.
        
        Args:
            training_config_path: Ruta al archivo training_config.yaml
        """
        with open(training_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model']['base_model']
        self.model = None
        self.results = None

    def setup_model(self):
        """Descarga el modelo base y lo inicializa."""
        print(f"\n1️⃣  Configurando modelo: {self.model_name}")
        try:
            model_path = ModelDownloader.download_model(self.model_name, verbose=True)
            self.model = YOLO(model_path)
            print("   ✅ Modelo cargado exitosamente.")
        except Exception as e:
            print(f"   ❌ Error configurando el modelo: {e}")
            raise

    def train(self, callbacks=None):
        """
        Ejecuta el ciclo de entrenamiento completo con los parámetros del archivo de configuración.
        """
        if not self.model:
            print("   ❌ Error: El modelo no ha sido inicializado. Ejecute `setup_model()` primero.")
            return

        print("\n2️⃣  Iniciando entrenamiento con YOLOv11-Pose...")
        
        # Cargar parámetros desde la configuración
        train_cfg = self.config['training']
        aug_cfg = self.config['augmentation']
        loss_cfg = self.config['loss_weights']
        paths_cfg = self.config['paths']
        
        self.results = self.model.train(
            data=paths_cfg['data_yaml'],
            epochs=train_cfg['epochs'],
            batch=train_cfg['batch_size'],
            imgsz=self.config['model']['input_size'],
            device=train_cfg['device'],
            workers=train_cfg['workers'],
            patience=train_cfg['patience'],
            
            # Ponderación de la función de pérdida
            box=loss_cfg['box'],
            cls=loss_cfg['cls'],
            dfl=loss_cfg['dfl'],
            pose=loss_cfg['pose'],
            kobj=loss_cfg['kobj'],
            
            # Aumento de datos
            hsv_h=aug_cfg['hsv_h'], hsv_s=aug_cfg['hsv_s'], hsv_v=aug_cfg['hsv_v'],
            degrees=aug_cfg['degrees'], translate=aug_cfg['translate'],
            scale=aug_cfg['scale'], shear=aug_cfg['shear'],
            perspective=aug_cfg['perspective'], flipud=aug_cfg['flipud'],
            fliplr=aug_cfg['fliplr'], mosaic=aug_cfg['mosaic'],
            
            # Rutas y nombres del proyecto
            project=paths_cfg['output_dir'],
            name='salmon_pose_yolov11',  # Nombre de corrida específico para v11
            exist_ok=True, # Permitir re-ejecutar en la misma carpeta
            
            # Callbacks (si se proporcionan)
            callbacks=callbacks
        )
        
        print("\n✅ Entrenamiento con YOLOv11 completado.")
        return self.results
