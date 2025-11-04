"""
Script principal de entrenamiento
"""
import yaml
from pathlib import Path
from ultralytics import YOLO
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.callbacks.custom_metrics_callback import CustomMetricsCallback


def main():
    print("=" * 80)
    print("🚀 ENTRENAMIENTO YOLOv8-POSE PARA SALMONES")
    print("=" * 80)
    
    # Cargar configuraciones
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n📁 Dataset: {config['paths']['data_yaml']}")
    print(f"🎯 Modelo base: {config['model']['base_model']}")
    print(f"📊 Épocas: {config['training']['epochs']}")
    print(f"🖼️  Resolución: {config['model']['input_size']}")
    
    # Inicializar modelo
    print("\n⏳ Cargando modelo...")
    model = YOLO(config['model']['base_model'])
    
    # Registrar callback
    callback = CustomMetricsCallback('config/keypoints_config.yaml')
    model.add_callback("on_val_end", callback.on_val_end)
    model.add_callback("on_train_end", callback.on_train_end)
    
    # Entrenar
    print("\n🔄 Iniciando entrenamiento...\n")
    results = model.train(
        data=config['paths']['data_yaml'],
        epochs=config['training']['epochs'],
        imgsz=config['model']['input_size'],
        batch=config['training']['batch_size'],
        patience=config['training']['patience'],
        device=config['training']['device'],
        workers=config['training']['workers'],
        
        # Loss weights
        box=config['training']['loss_weights']['box'],
        cls=config['training']['loss_weights']['cls'],
        dfl=config['training']['loss_weights']['dfl'],
        pose=config['training']['loss_weights']['pose'],
        kobj=config['training']['loss_weights']['kobj'],
        
        # Augmentations
        **config['augmentation'],
        
        # Output
        project=config['paths']['output_dir'],
        name='salmon_pose_v1',
        exist_ok=False,
        verbose=True,
        save=True,
        plots=True
    )
    
    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"\n📂 Resultados guardados en:")
    print(f"   {results.save_dir}")
    print(f"\n📊 Métricas disponibles en:")
    print(f"   results.csv - Métricas automáticas por época")


if __name__ == '__main__':
    main()
