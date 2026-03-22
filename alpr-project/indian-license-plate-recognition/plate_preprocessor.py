"""
Advanced image preprocessing for license plate recognition
Optimized techniques to improve OCR accuracy
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from logger_util import setup_logger
from config_manager import config

logger = setup_logger(__name__)


class PlatePreprocessor:
    """Advanced preprocessing for license plate images"""
    
    def __init__(self):
        """Initialize preprocessor with config"""
        self.cfg = config.preprocessing
    
    def preprocess_plate(self, image: np.ndarray, 
                        use_all_methods: bool = False) -> np.ndarray:
        """
        Main preprocessing pipeline for best results
        
        Args:
            image: Input BGR image
            use_all_methods: If True, returns best result from multiple methods
        
        Returns:
            Preprocessed image for OCR
        """
        if image is None or image.size == 0:
            logger.error("Invalid image input")
            return None
        
        # Convert to grayscale
        gray = self._ensure_grayscale(image)
        
        if use_all_methods:
            # Try multiple methods and use the one with best contrast
            methods = {
                'clahe': self._clahe_enhancement(gray),
                'bilateral': self._bilateral_filtering(gray),
                'adaptive': self._adaptive_thresholding(gray),
            }
            
            # Return method with best contrast
            return self._select_best_preprocessing(methods)
        else:
            # Use optimized single pipeline
            return self._optimized_pipeline(gray)
    
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()
    
    def _optimized_pipeline(self, gray: np.ndarray) -> np.ndarray:
        """
        Optimized single preprocessing pipeline
        Best balance between quality and speed
        """
        # Step 1: Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=self.cfg.clahe_tile_size
        )
        enhanced = clahe.apply(gray)
        
        # Step 2: Bilateral filtering to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(
            enhanced,
            d=self.cfg.bilateral_diameter,
            sigmaColor=self.cfg.bilateral_sigma_color,
            sigmaSpace=self.cfg.bilateral_sigma_space
        )
        
        # Step 3: Otsu's thresholding for binary image
        _, binary = cv2.threshold(
            filtered, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Step 4: Morphological operations to clean up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.cfg.morph_kernel_size
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _clahe_enhancement(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=self.cfg.clahe_tile_size
        )
        return clahe.apply(gray)
    
    def _bilateral_filtering(self, gray: np.ndarray) -> np.ndarray:
        """Advanced bilateral filtering for edge-preserving smoothing"""
        # Apply CLAHE first
        clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=self.cfg.clahe_tile_size
        )
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            enhanced,
            d=self.cfg.bilateral_diameter,
            sigmaColor=self.cfg.bilateral_sigma_color,
            sigmaSpace=self.cfg.bilateral_sigma_space
        )
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return binary
    
    def _adaptive_thresholding(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for varying lighting conditions"""
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Morphological closing to fill gaps
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.cfg.morph_kernel_size
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _select_best_preprocessing(self, methods: Dict[str, np.ndarray]) -> np.ndarray:
        """Select preprocessing method with best contrast"""
        best_method = None
        best_contrast = -1
        
        for method_name, processed_img in methods.items():
            # Calculate Laplacian variance (measure of contrast)
            contrast = cv2.Laplacian(processed_img, cv2.CV_64F).var()
            
            if contrast > best_contrast:
                best_contrast = contrast
                best_method = processed_img
        
        logger.debug(f"Selected preprocessing with contrast: {best_contrast:.2f}")
        return best_method if best_method is not None else methods['clahe']
    
    def resize_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to optimal dimensions for OCR
        License plates should be between 50-100 pixels in height
        """
        h, w = image.shape[:2]
        
        # If image is too small
        if h < self.cfg.min_plate_height:
            scale = self.cfg.min_plate_height / h
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, self.cfg.min_plate_height),
                             interpolation=cv2.INTER_CUBIC)
        
        # If image is too large
        elif h > self.cfg.max_plate_height:
            scale = self.cfg.max_plate_height / h
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, self.cfg.max_plate_height),
                             interpolation=cv2.INTER_AREA)
        
        return image
    
    def correct_rotation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct plate rotation
        
        Returns:
            Tuple of (corrected_image, rotation_angle)
        """
        gray = self._ensure_grayscale(image)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough line detection to find plate orientation
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        
        if lines is None or len(lines) == 0:
            return image, 0.0
        
        # Extract angles and find most common one
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            # Convert to -90 to 90 range
            if angle > 90:
                angle = angle - 180
            angles.append(angle)
        
        # Use median angle for robustness
        median_angle = np.median(angles)
        
        if abs(median_angle) < 2:  # Minimal rotation
            return image, median_angle
        
        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        
        logger.debug(f"Corrected rotation: {median_angle:.2f}°")
        return rotated, median_angle
    
    def enhance_contrast(self, image: np.ndarray, 
                        contrast_factor: float = 1.5) -> np.ndarray:
        """
        Enhance image contrast using CLAHE
        
        Args:
            image: Input image
            contrast_factor: Factor to apply (1.0 = no change, >1.0 = more contrast)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=contrast_factor * 2.0,
            tileGridSize=(8, 8)
        )
        return clahe.apply(image)
    
    def reduce_noise(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Reduce image noise while preserving edges
        
        Args:
            image: Input image
            method: 'bilateral', 'morphological', or 'gaussian'
        """
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 11, 17, 17)
        
        elif method == 'morphological':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            return closing
        
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        
        else:
            logger.warning(f"Unknown noise reduction method: {method}")
            return image


# Example usage
if __name__ == "__main__":
    preprocessor = PlatePreprocessor()
    
    # Example: Load and preprocess an image
    test_image_path = "sample_plate.jpg"
    
    try:
        image = cv2.imread(test_image_path)
        
        if image is not None:
            # Preprocess
            processed = preprocessor.preprocess_plate(image)
            
            # Correct rotation
            rotated, angle = preprocessor.correct_rotation(processed)
            
            # Resize for OCR
            resized = preprocessor.resize_for_ocr(rotated)
            
            # Display results
            cv2.imshow("Original", image)
            cv2.imshow("Processed", processed)
            cv2.imshow("Resized", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Could not load test image")
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
