# Contenido para: scripts/01_train.py
"""
Script principal para lanzar el entrenamiento de YOLOv11-Pose.
"""
import sys
from pathlib import Path

# Añadir el directorio 'src' al path para las importaciones
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar las clases necesarias
from src.models.yolo_wrapper import YOLOv11PoseTrainer
from src.callbacks.custom_metrics_callback import CustomMetricsCallback

def run_training():
    """
    Orquesta el proceso completo de entrenamiento para YOLOv11.
    """
    print("\n" + "="*80)
    print("🚀 INICIANDO PIPELINE DE ENTRENAMIENTO PARA YOLOv11-POSE")
    print("="*80)
    
    config_path = 'configs/training_config.yaml'
    keypoints_config_path = 'configs/keypoints_config.yaml'

    # 1. Instanciar el Trainer de YOLOv11
    try:
        trainer = YOLOv11PoseTrainer(config_path)
        trainer.setup_model()
    except Exception as e:
        print(f"\n❌ Error Crítico al inicializar el modelo: {e}")
        return

    # 2. Instanciar y registrar el Callback de Métricas
    print("\n3️⃣  Configurando métricas personalizadas (OKS, PCK)...")
    try:
        custom_metrics = CustomMetricsCallback(keypoints_config_path)
        print("   ✅ Callback de métricas listo.")
    except Exception as e:
        print(f"\n❌ Error Crítico al inicializar las métricas: {e}")
        return
        
    # 3. Iniciar el entrenamiento
    trainer.train(callbacks=custom_metrics.get_callbacks())

    print("\n" + "="*80)
    print("🎉 PIPELINE DE ENTRENAMIENTO FINALIZADO")
    print("="*80)

if __name__ == '__main__':
    run_training()
