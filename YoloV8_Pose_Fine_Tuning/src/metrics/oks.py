"""
Implementación de OKS (Object Keypoint Similarity)
"""
import numpy as np


class OKSMetric:
    """Calcula OKS siguiendo el estándar COCO"""
    
    def __init__(self, sigmas: np.ndarray):
        """
        Args:
            sigmas: (K,) - constantes kappa por keypoint
        """
        self.sigmas = np.array(sigmas)
        
    def compute(
        self,
        pred_keypoints: np.ndarray,
        gt_keypoints: np.ndarray,
        bbox: np.ndarray,
        visibility: np.ndarray
    ) -> float:
        """
        Calcula OKS para una detección
        
        Args:
            pred_keypoints: (K, 2) - keypoints predichos
            gt_keypoints: (K, 2) - keypoints ground truth
            bbox: (4,) - bounding box [x1, y1, x2, y2]
            visibility: (K,) - visibilidad por keypoint
        
        Returns:
            oks: Object Keypoint Similarity [0, 1]
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        scale = np.sqrt(area)
        
        distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
        
        oks_per_kpt = np.exp(
            -(distances**2) / (2 * scale**2 * self.sigmas**2)
        )
        
        valid_mask = visibility > 0
        
        if valid_mask.sum() == 0:
            return 0.0
        
        oks = oks_per_kpt[valid_mask].sum() / valid_mask.sum()
        return oks
    
    def compute_batch(
        self,
        pred_keypoints: np.ndarray,
        gt_keypoints: np.ndarray,
        bboxes: np.ndarray,
        visibilities: np.ndarray
    ) -> np.ndarray:
        """
        Calcula OKS para un batch
        """
        N = pred_keypoints.shape[0]
        oks_scores = np.zeros(N)
        
        for i in range(N):
            oks_scores[i] = self.compute(
                pred_keypoints[i],
                gt_keypoints[i],
                bboxes[i],
                visibilities[i]
            )
        
        return oks_scores
