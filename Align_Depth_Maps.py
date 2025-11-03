import numpy as np
import cv2
import glob
import os

def align_depth_maps(rgb_folder, depth_folder, output_folder_npy, output_folder_png=None):
    """
    Alinea mapas de profundidad con imágenes RGB y guarda resultados en .npy y opcionalmente en PNG.
    """
    
    # Crear carpetas de salida si no existen
    if not os.path.exists(output_folder_npy):
        os.makedirs(output_folder_npy)
    if output_folder_png is not None and not os.path.exists(output_folder_png):
        os.makedirs(output_folder_png)
    
    # Obtener listas de archivos
    rgb_files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_folder, "*.npy")))
    
    # Verificar que tenemos el mismo número de archivos
    if len(rgb_files) != len(depth_files):
        print(f"Error: Número diferente de archivos RGB ({len(rgb_files)}) y Depth ({len(depth_files)})")
        print("Asegúrate de que los archivos tengan nombres correspondientes")
        return
    
    print(f"Encontrados {len(rgb_files)} pares de imágenes")
    
    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        print(f"Procesando par {i+1}/{len(rgb_files)}")
        
        # Cargar imágenes
        rgb = cv2.imread(rgb_file)
        depth_data = np.load(depth_file)
        
        if rgb is None or depth_data is None:
            print(f"Error cargando: {rgb_file} o {depth_file}")
            continue
        
        # Alinear mapas de profundidad
        aligned_depth = align_depth_to_rgb(rgb, depth_data)
        
        # Guardar resultado en .npy
        filename_base = os.path.basename(depth_file).replace('.npy', '')
        npy_save_path = os.path.join(output_folder_npy, filename_base + '_aligned.npy')
        np.save(npy_save_path, aligned_depth)
        print(f"Guardado .npy: {npy_save_path}")
        
        # Guardar como imagen PNG (normalizada) si se especifica la carpeta
        if output_folder_png is not None:
            # Normalizar a 8-bit para visualización
            depth_min = np.min(aligned_depth)
            depth_max = np.max(aligned_depth)
            if depth_max > depth_min:
                depth_img = ((aligned_depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_img = np.zeros(aligned_depth.shape, dtype=np.uint8)
            
            # Aplicar colormap para mejor visualización (opcional)
            depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
            
            png_save_path = os.path.join(output_folder_png, filename_base + '_aligned.png')
            cv2.imwrite(png_save_path, depth_colored)
            print(f"Guardado PNG: {png_save_path}")

def align_depth_to_rgb(rgb, depth_data):
    """
    Alinea un mapa de profundidad a una imagen RGB usando múltiples técnicas
    """
    
    # Convertir RGB a escala de grises para el procesamiento
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    
    # Normalizar depth data para procesamiento
    depth_normalized = normalize_depth_data(depth_data)
    
    # Método 1: Registro basado en características (SIFT/ORB)
    transformation = feature_based_alignment(rgb_gray, depth_normalized)
    
    # Aplicar transformación si se encontró
    if transformation is not None:
        if transformation.shape == (3, 3):  # Homografía
            aligned_depth = apply_homography_to_depth(depth_data, transformation, rgb.shape)
        else:  # Transformación afín
            aligned_depth = apply_affine_to_depth(depth_data, transformation, rgb.shape)
    else:
        # Si ambos métodos fallan, usar el depth original
        print("  Métodos de alineamiento fallaron, usando depth original")
        aligned_depth = depth_data
    
    return aligned_depth

def normalize_depth_data(depth_data):
    """
    Normaliza los datos de profundidad para procesamiento con OpenCV
    """
    # Normalizar a rango 0-255
    depth_min = np.min(depth_data)
    depth_max = np.max(depth_data)
    
    if depth_max > depth_min:
        depth_normalized = ((depth_data - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth_data, dtype=np.uint8)
    
    return depth_normalized

def feature_based_alignment(reference, moving):
    """
    Alineamiento basado en características (usa ORB como alternativa si SIFT no está disponible)
    """
    try:
        # Intentar usar SIFT primero
        try:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(reference, None)
            kp2, des2 = sift.detectAndCompute(moving, None)
            detector_type = "SIFT"
        except:
            # Fallback a ORB
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(reference, None)
            kp2, des2 = orb.detectAndCompute(moving, None)
            detector_type = "ORB"
        
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return None
        
        # Emparejar características
        if detector_type == "SIFT":
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Aplicar ratio test de Lowe
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        else:  # ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:50]
        
        if len(good_matches) < 4:
            return None
        
        # Obtener puntos correspondientes
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Calcular homografía
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        return H
        
    except Exception as e:
        print(f"  Error en alineamiento por características: {e}")
        return None

def apply_homography_to_depth(depth_data, homography, target_shape):
    """
    Aplica homografía a los datos de profundidad
    """
    height, width = target_shape[:2]
    
    # Aplicar transformación
    aligned_depth = cv2.warpPerspective(depth_data, homography, (width, height),
                                       flags=cv2.INTER_NEAREST)  # Usar INTER_NEAREST para datos de profundidad
    
    return aligned_depth

def apply_affine_to_depth(depth_data, affine_matrix, target_shape):
    """
    Aplica transformación afín a los datos de profundidad
    """
    height, width = target_shape[:2]
    
    # Aplicar transformación
    aligned_depth = cv2.warpAffine(depth_data, affine_matrix, (width, height),
                                  flags=cv2.INTER_NEAREST)  # Usar INTER_NEAREST para datos de profundidad
    
    return aligned_depth

def phase_correlation_alignment(reference, moving):
    """
    Alineamiento por correlación de fase usando solo numpy y OpenCV
    """
    try:
        # Aplicar transformada de Fourier
        ref_fft = np.fft.fft2(reference)
        mov_fft = np.fft.fft2(moving)
        
        # Calcular correlación cruzada
        cross_power = (ref_fft * np.conj(mov_fft)) / (np.abs(ref_fft * np.conj(mov_fft)) + 1e-10)
        
        # Transformada inversa
        correlation = np.fft.ifft2(cross_power)
        correlation = np.abs(correlation)
        
        # Encontrar pico de correlación
        y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Ajustar desplazamiento si es mayor que la mitad de la imagen
        if y_shift > reference.shape[0] // 2:
            y_shift -= reference.shape[0]
        if x_shift > reference.shape[1] // 2:
            x_shift -= reference.shape[1]
        
        # Crear matriz de transformación afín
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        
        return M
        
    except Exception as e:
        print(f"  Error en alineamiento por correlación de fase: {e}")
        return None

def main():
    # Configurar rutas
    rgb_folder = "Dataset_PDI/output/"
    depth_folder = "Dataset_PDI/depth/"  # Carpeta con archivos .npy
    output_folder_npy = "Dataset_PDI/aligned_depth_npy/"  # Para guardar .npy
    output_folder_png = "Dataset_PDI/aligned_depth_png/"  # Para guardar .png (visualización)
    
    # Ejecutar alineamiento
    align_depth_maps(rgb_folder, depth_folder, output_folder_npy, output_folder_png)
    print("Proceso de alineamiento completado")

if __name__ == "__main__":
    main()