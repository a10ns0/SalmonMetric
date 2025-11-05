"""
Implementación de PCK (Percentage of Correct Keypoints)
"""
import numpy as np
from typing import Tuple


class PCKMetric:
    """Calcula PCK para pose estimation"""
    
    def __init__(self, threshold: float = 0.2, normalize: bool = True):
        """
        Args:
            threshold: Umbral de distancia normalizado (default 0.2)
            normalize: Si True, normaliza por diagonal del bbox
        """
        self.threshold = threshold
        self.normalize = normalize
        
    def compute(
        self, 
        pred_keypoints: np.ndarray,
        gt_keypoints: np.ndarray,
        bbox: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calcula PCK
        
        Args:
            pred_keypoints: (N, K, 2) - keypoints predichos
            gt_keypoints: (N, K, 2) - keypoints ground truth
            bbox: (N, 4) - bounding boxes [x1, y1, x2, y2]
        
        Returns:
            pck_global: PCK promedio (%)
            pck_per_keypoint: PCK por keypoint (K,)
        """
        assert pred_keypoints.shape == gt_keypoints.shape, \
            f"Shape mismatch: {pred_keypoints.shape} vs {gt_keypoints.shape}"
        
        N, K, _ = pred_keypoints.shape
        
        if self.normalize:
            bbox_diag = np.sqrt(
                (bbox[:, 2] - bbox[:, 0])**2 + 
                (bbox[:, 3] - bbox[:, 1])**2
            )
            threshold_dist = bbox_diag[:, None] * self.threshold
        else:
            threshold_dist = np.ones((N, 1)) * self.threshold
        
        distances = np.linalg.norm(
            pred_keypoints - gt_keypoints, 
            axis=2
        )
        
        correct = distances < threshold_dist
        pck_global = (correct.sum() / correct.size) * 100
        pck_per_keypoint = (correct.sum(axis=0) / N) * 100
        
        return pck_global, pck_per_keypoint
