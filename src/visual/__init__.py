"""
Visual Analysis Module for BIM Inspection

Provides CLIP-based visual alignment between site photos and BIM element descriptions.
Used for matching defect images to corresponding IFC elements.
"""

from .aligner import VisualAligner
from .image_parser import ImageParserReader

__all__ = ["VisualAligner", "ImageParserReader"]
