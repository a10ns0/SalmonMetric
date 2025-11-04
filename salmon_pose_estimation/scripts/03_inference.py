"""
Script de inferencia en imágenes
"""
import argparse
import cv2
from ultralytics import YOLO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Inferencia YOLOv8-Pose')
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen')
    parser.add_argument('--model', type=str, default='outputs/runs/salmon_pose_v1/weights/best.pt',
                        help='Ruta al modelo entrenado')
    parser.add_argument('--conf', type=float, default=0.5, help='Confianza mínima')
    parser.add_argument('--output', type=str, default='outputs/inference_result.jpg',
                        help='Ruta para guardar resultado')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 INFERENCIA YOLOv8-POSE")
    print("=" * 80)
    
    # Cargar modelo
    print(f"\n🔍 Cargando modelo: {args.model}")
    model = YOLO(args.model)
    
    # Cargar imagen
    print(f"📷 Cargando imagen: {args.image}")
    image = cv2.imread(args.image)
    
    if image is None:
        print(f"❌ Error: No se pudo cargar la imagen {args.image}")
        return
    
    # Inferencia
    print(f"\n⏳ Realizando inferencia...")
    results = model(image, conf=args.conf)
    
    # Procesar resultados
    for i, result in enumerate(results):
        if len(result.boxes) > 0:
            print(f"\n✅ Detecciones encontradas: {len(result.boxes)}")
            
            for j, box in enumerate(result.boxes):
                print(f"\n  Salmón {j+1}:")
                print(f"    Confianza: {box.conf[0]:.3f}")
                
                if hasattr(result, 'keypoints'):
                    kpts = result.keypoints
                    print(f"    Keypoints detectados: {len(kpts.xy[j])}")
        else:
            print("❌ No se encontraron salmones en la imagen")
    
    # Guardar resultado
    annotated_image = results[0].plot()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated_image)
    
    print(f"\n💾 Resultado guardado en: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
