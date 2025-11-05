
# Sistema de Estimación de Pose en Salmones con YOLOv11 (Salmon-Pose-Estimation-v11)

Este proyecto implementa un pipeline completo para el *fine-tuning* de modelos **YOLOv11-Pose** con el objetivo de localizar con precisión 11 puntos anatómicos clave en salmones. El sistema está diseñado para ser una réplica modular del proyecto YOLOv8, permitiendo una comparación directa y justa entre ambas arquitecturas de modelo.

## 1. Características Principales

* **Arquitectura Modular**: El código está organizado en módulos cohesivos con responsabilidades únicas (modelos, métricas, callbacks, etc.).
* **Configuración Centralizada**: Todo el comportamiento del pipeline (hiperparámetros, rutas, etc.) se controla desde archivos YAML, permitiendo experimentar sin modificar el código.
* **Métricas Personalizadas**: Integra métricas avanzadas de pose como OKS (Object Keypoint Similarity) y PCK (Percentage of Correct Keypoints) directamente en el ciclo de validación de `ultralytics`.
* **Pipeline Automatizado**: Incluye scripts para automatizar todo el flujo de trabajo: preparación del dataset, entrenamiento, evaluación e inferencia.

## 2. Estructura del Proyecto

La organización del repositorio separa claramente la lógica del núcleo (`src`), la configuración (`configs`), los datos (`data`) y los scripts de ejecución (`scripts`).

```
YoloV11_Pose/
├── configs/                # Archivos de configuración (hiperparámetros, etc.)
├── data/                   # Dataset procesado y listo para YOLO (generado automáticamente)
├── dataset_fuente/         # Directorio sugerido para TUS datos originales
├── notebooks/              # Jupyter Notebooks para exploración y análisis
├── outputs/                # Directorio para guardar resultados (modelos, métricas, logs)
├── scripts/                # Scripts para ejecutar el pipeline (setup, train, eval, infer)
├── src/                    # Código fuente principal de la aplicación
│   ├── callbacks/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── tests/                  # Pruebas unitarias para asegurar la calidad del código
├── README.md
└── requirements.txt        # Dependencias del proyecto
```

## 3. Cómo Empezar

### 3.1. Prerrequisitos

* Python 3.8 o superior
* PyTorch con soporte para CUDA (recomendado)
* Versión de `ultralytics` compatible con YOLOv11 (>=8.1.0)

### 3.2. Instalación

1. Clona este repositorio:

   ```
   gh repo clone a10ns0/SalmonMetric
   cd YoloV11_Pose
   ```
2. Crea un entorno virtual e instala las dependencias:

   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### 3.3. Flujo de Trabajo

El pipeline se ejecuta a través de los scripts ubicados en el directorio `scripts/`.

#### Paso 1: Configuración del Entorno y Preparación del Dataset

Antes de empezar, **coloca todas tus imágenes (`.jpg`, `.png`) y sus correspondientes archivos de etiquetas (`.txt`) en una única carpeta de origen**.

El script `00_setup.py` es multifuncional. Su tarea principal es preparar todo el entorno. Opcionalmente, si le proporcionas la ruta a tu carpeta de datos, también procesará el dataset: lo dividirá en conjuntos de entrenamiento, validación y prueba, y generará el archivo `data/data.yaml` que YOLO necesita.

Ejecuta el script desde la raíz del proyecto.

**Para configurar el entorno Y procesar tu dataset:**

```
# Ejemplo: si tus datos están en una carpeta llamada 'mis_datos_anotados'
python scripts/00_setup.py --source-dir mis_datos_anotados
```

*Nota: Para una comparación justa con el proyecto de YOLOv8, puedes apuntar al mismo directorio de datos de origen.*

**Para configurar SOLO el entorno (sin procesar datos):**

```
# Omite el argumento --source-dir
python scripts/00_setup.py
```

#### Paso 2: Entrenamiento del Modelo YOLOv11

Una vez que el dataset ha sido procesado y se encuentra en la carpeta `data/`, lanza el script de entrenamiento específico para YOLOv11.

```
python scripts/01_train.py
```

Este script cargará la configuración desde `configs/training_config.yaml` (asegúrate de que `base_model` apunte a `yolov11s-pose.pt` u otra variante de v11) y comenzará el `fine-tuning`. Los resultados se guardarán en una carpeta específica, como `outputs/runs/YoloV11_Pose`.

#### Paso 3: Evaluación del Modelo

Utiliza el mejor modelo guardado (`best.pt`) para realizar una evaluación formal sobre el conjunto de prueba.

```
python scripts/02_evaluate.py
```

#### Paso 4: Inferencia

Ejecuta una inferencia de ejemplo en una imagen o video para visualizar las predicciones del modelo.

```
python scripts/03_inference.py
```

## 4. Arquitectura Detallada

### `src` - El Corazón del Proyecto

* **`src/models/yolo_wrapper.py`**: Abstrae el proceso de entrenamiento, **configurado para usar modelos YOLOv11**.
* **`src/metrics/`**: Módulos para **OKS** y **PCK**. Son idénticos a los del proyecto YOLOv8, garantizando que la comparación de métricas sea directa y justa.
* **`src/callbacks/custom_metrics_callback.py`**: Puente entre el motor de `ultralytics` y nuestro sistema de métricas. Funciona de la misma manera para ambas versiones de YOLO.
* **`src/utils/download_utils.py`**: Utilidad para gestionar la descarga de los modelos base de YOLO, incluyendo las nuevas variantes de YOLOv11.

### `configs` - El Cerebro del Proyecto

* **`training_config.yaml`**: Archivo de configuración principal. Para este proyecto, la clave `model.base_model` debe apuntar a una variante de YOLOv11 (ej. `"yolov11s-pose.pt"`). Para una comparación justa, el resto de los hiperparámetros (tamaño de imagen, épocas, etc.) deben ser idénticos a los del proyecto YOLOv8.
* **`keypoints_config.yaml`**: Define la semántica de la tarea (nombres de keypoints y sigmas de OKS). Es **compartido e idéntico** al del proyecto YOLOv8 para asegurar una evaluación consistente.

## 5. Comparación con YOLOv8

El propósito principal de este proyecto es servir como un gemelo experimental del proyecto `salmon_pose_estimation_yolov8`. Al mantener la misma estructura de datos, configuración de entrenamiento y código de evaluación, podemos aislar el impacto de la arquitectura del modelo (YOLOv11 vs. YOLOv8) en la precisión y la velocidad.

Un script de comparación (ej. `scripts/04_compare.py` en el directorio padre) puede ser utilizado para leer los resultados de ambos proyectos y generar un reporte cuantitativo del rendimiento.

```

```
