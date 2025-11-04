"""
Salmon Pose Estimation Package
"""
__version__ = "1.0.0"
__author__ = "Your Name"

from .metrics import PoseEvaluator, PCKMetric, OKSMetric
from .callbacks import CustomMetricsCallback

__all__ = [
    'PoseEvaluator',
    'PCKMetric',
    'OKSMetric',
    'CustomMetricsCallback',
]
