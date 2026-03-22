"""
Configuration management for Indian License Plate Recognition System
Centralizes all settings and makes them easy to modify
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DetectionConfig:
    """Detection-related configuration"""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.4
    device: str = "cpu"  # "cpu" or "cuda"
    
    # Fallback contour detection
    use_contour_fallback: bool = True
    contour_area_threshold: int = 1000
    min_aspect_ratio: float = 2.0
    max_aspect_ratio: float = 5.0


@dataclass
class OCRConfig:
    """OCR-related configuration"""
    languages: List[str] = None
    confidence_threshold: float = 0.4
    use_gpu: bool = False
    allowlist: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration"""
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple = (8, 8)
    
    # Bilateral filter
    bilateral_diameter: int = 11
    bilateral_sigma_color: int = 17
    bilateral_sigma_space: int = 17
    
    # Morphological operations
    morph_kernel_size: tuple = (2, 2)
    
    # Target plate dimensions for OCR
    min_plate_height: int = 50
    max_plate_height: int = 100
    
    # Dilate/erode iterations
    iterations: int = 1


@dataclass
class ValidationConfig:
    """License plate validation configuration"""
    # Indian license plate patterns
    patterns: List[str] = None
    
    # State codes (for validation)
    valid_state_codes: List[str] = None
    
    # Character confusion mapping for OCR correction
    char_confusion_map: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.patterns is None:
            self.patterns = [
                r'^[A-Z]{2}\s*\d{2}\s*[A-Z]{1,2}\s*\d{4}$',  # Standard
                r'^[A-Z]{2}\s*\d{2}\s*[A-Z]{1}\s*\d{4}$',    # Old format
                r'^[A-Z]{2}\s*\d{2}\s*[A-Z]{2}\s*\d{1,4}$',  # Variable digits
            ]
        
        if self.valid_state_codes is None:
            self.valid_state_codes = [
                'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK',
                'JH', 'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'OD', 'PB',
                'RJ', 'SK', 'TN', 'TG', 'TR', 'UP', 'UK', 'WB', 'DL', 'CH',
                'DD', 'DN', 'LD', 'PY'
            ]
        
        if self.char_confusion_map is None:
            self.char_confusion_map = {
                'H': ['8', 'B'], 'O': ['0', 'Q'], 'I': ['1', 'l'],
                'S': ['5'], 'Z': ['2'], 'G': ['6'], 'D': ['0'],
                'B': ['8', '6'], 'R': ['8'], 'A': ['4'], 'E': ['3'],
                'T': ['7'], '0': ['O', 'D', 'Q'], '1': ['I', 'l'],
                '2': ['Z'], '3': ['E'], '4': ['A'], '5': ['S'],
                '6': ['G', 'B'], '7': ['T'], '8': ['B', 'H', 'R'],
            }


@dataclass
class OutputConfig:
    """Output and visualization configuration"""
    save_results: bool = True
    output_dir: str = "results"
    save_annotated_images: bool = True
    save_csv: bool = True
    draw_confidence_scores: bool = True
    draw_bounding_boxes: bool = True
    box_color: tuple = (0, 255, 0)  # BGR format
    text_color: tuple = (0, 255, 0)


class Config:
    """Main configuration class that combines all sub-configs"""
    
    def __init__(self, config_file: str = None):
        self.detection = DetectionConfig()
        self.ocr = OCRConfig()
        self.preprocessing = PreprocessingConfig()
        self.validation = ValidationConfig()
        self.output = OutputConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, filename: str):
        """Load configuration from JSON file"""
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
                self._update_from_dict(config_dict)
            print(f"✅ Config loaded from {filename}")
        except Exception as e:
            print(f"⚠️ Failed to load config from {filename}: {e}")
    
    def save_to_file(self, filename: str):
        """Save configuration to JSON file"""
        try:
            config_dict = {
                'detection': self._dataclass_to_dict(self.detection),
                'ocr': self._dataclass_to_dict(self.ocr),
                'preprocessing': self._dataclass_to_dict(self.preprocessing),
                'validation': self._dataclass_to_dict(self.validation),
                'output': self._dataclass_to_dict(self.output),
            }
            
            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"✅ Config saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save config: {e}")
    
    def _dataclass_to_dict(self, obj) -> Dict:
        """Convert dataclass to dictionary"""
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, (list, str, int, float, bool, tuple)):
                result[key] = value
            elif isinstance(value, dict):
                result[key] = value
        return result
    
    def _update_from_dict(self, config_dict: Dict):
        """Update configuration from dictionary"""
        if 'detection' in config_dict:
            for key, value in config_dict['detection'].items():
                if hasattr(self.detection, key):
                    setattr(self.detection, key, value)
        
        if 'ocr' in config_dict:
            for key, value in config_dict['ocr'].items():
                if hasattr(self.ocr, key):
                    setattr(self.ocr, key, value)
        
        if 'preprocessing' in config_dict:
            for key, value in config_dict['preprocessing'].items():
                if hasattr(self.preprocessing, key):
                    setattr(self.preprocessing, key, value)
        
        if 'output' in config_dict:
            for key, value in config_dict['output'].items():
                if hasattr(self.output, key):
                    setattr(self.output, key, value)


# Global config instance
config = Config()
