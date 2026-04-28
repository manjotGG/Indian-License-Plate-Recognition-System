"""
Refactored Indian License Plate Recognition System
Modular, production-ready architecture with improved error handling
"""

import cv2
import numpy as np
import easyocr
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from ..utils.config import config
from ..utils.logger import setup_logger
from .validator import IndianPlateValidator
from .preprocessor import PlatePreprocessor
from ..utils.database import ALPRDatabase

logger = setup_logger(__name__, log_file="alpr.log")


@dataclass
class Detection:
    """Single detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    method: str  # 'YOLO' or 'Contour'


@dataclass
class OCRResult:
    """OCR result for a plate"""
    text: str
    confidence: float
    validated: bool
    state_code: Optional[str] = None
    registration_district: Optional[str] = None
    series: Optional[str] = None
    sequential_number: Optional[str] = None


@dataclass
class PlateResult:
    """Complete result for one detected plate"""
    detection: Detection
    ocr_result: OCRResult
    processing_time: float
    image_roi: Optional[np.ndarray] = None


class ImprovedIndianLPRSystem:
    """
    Production-ready Indian License Plate Recognition System
    Modular architecture with proper error handling
    """
    
    def __init__(self, device: str = 'cpu', use_gpu: bool = False, enable_database: bool = False):
        """
        Initialize the LPR system
        
        Args:
            device: 'cpu' or 'cuda'
            use_gpu: Force GPU usage
            enable_database: Enable database storage
        """
        self.device = self._setup_device(device, use_gpu)
        self.model = None
        self.reader = None
        self.validator = IndianPlateValidator()
        self.preprocessor = PlatePreprocessor()
        self.database = ALPRDatabase() if enable_database else None
        
        logger.info(f"Initializing ALPR System on {self.device}")
        self._load_models()
    
    def _setup_device(self, device: str, use_gpu: bool) -> str:
        """Setup compute device"""
        if use_gpu:
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info("✅ GPU available, using CUDA")
            elif torch.backends.mps.is_available():
                device = 'mps'
                logger.info("✅ Apple Silicon GPU available, using MPS")
            else:
                logger.warning("⚠️ GPU requested but not available, falling back to CPU")
                device = 'cpu'
        else:
            device = 'cpu'
        
        return device
    
    def _load_models(self):
        """Load YOLO and EasyOCR models"""
        # Load YOLO
        try:
            from ultralytics import YOLO
            model_path = config.detection.model_path
            
            # Try multiple possible paths for the model
            possible_paths = [
                model_path,
                Path(__file__).parent.parent.parent / model_path,  # From src/core/ to project root
                Path(__file__).parent.parent.parent / "models" / "license_plate_yolov8.pt",
            ]
            
            loaded_model = None
            for path in possible_paths:
                if Path(path).exists():
                    self.model = YOLO(str(path))
                    logger.info(f"✅ YOLO model loaded: {path}")
                    loaded_model = self.model
                    break
            
            if loaded_model is None:
                logger.warning(f"⚠️ Model file not found at any of: {possible_paths}")
                logger.warning("⚠️ Will use fallback contour detection")
                self.model = None
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            logger.warning("⚠️ Will use fallback contour detection")
            self.model = None
        
        # Load EasyOCR
        try:
            # EasyOCR GPU support: only CUDA, not MPS
            use_gpu_ocr = self.device == 'cuda'
            self.reader = easyocr.Reader(
                config.ocr.languages,
                gpu=use_gpu_ocr
            )
            logger.info(f"✅ EasyOCR initialized (GPU: {use_gpu_ocr})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize EasyOCR: {e}")
            self.reader = None
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image from disk
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with processing results
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        return self.process_image_from_array(image, filename=str(image_path))

    def process_image_from_array(self, image: np.ndarray, filename: str = "uploaded.jpg") -> Dict:
        """
        Process an image provided as a NumPy array
        
        Args:
            image: BGR image array
            filename: Original filename or identifier
        
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()
        result = {
            'image_path': filename,
            'status': 'success',
            'message': '',
            'plates': [],
            'processing_time': 0,
            'timestamp': datetime.now().isoformat(),
        }

        try:
            if image is None or image.size == 0:
                raise ValueError("Provided image array is empty or invalid")

            logger.info(f"Processing image array: {filename}")
            
            plate_results = self.detect_and_recognize(image)
            
            for plate in plate_results:
                result['plates'].append({
                    'plate_text': plate.ocr_result.text,
                    'confidence': plate.ocr_result.confidence,
                    'state': plate.ocr_result.state_code or '',
                    'district': plate.ocr_result.registration_district or '',
                    'series': plate.ocr_result.series or '',
                    'number': plate.ocr_result.sequential_number or '',
                    'bbox': plate.detection.bbox,
                })

            if not plate_results:
                result['message'] = "No license plates detected"
                logger.warning(result['message'])
            else:
                result['message'] = f"Successfully detected {len(plate_results)} plate(s)"
                logger.info(result['message'])

        except Exception as e:
            result['status'] = 'error'
            result['message'] = str(e)
            logger.error(f"Error processing image array: {e}", exc_info=True)

        finally:
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            if self.database:
                self.database.save_processing_session(result)

        return result

    def detect_and_recognize(self, image: np.ndarray) -> List[PlateResult]:
        """
        Main pipeline: detect plates and recognize text
        
        Args:
            image: BGR image
        
        Returns:
            List of PlateResult objects
        """
        results = []
        
        try:
            # Detect license plates
            detections = self._detect_plates(image)
            
            if not detections:
                logger.warning("No plates detected")
                return results
            
            logger.info(f"Detected {len(detections)} potential plate(s)")
            
            # Process each detection
            for detection in detections:
                plate_result = self._process_detection(image, detection)
                if plate_result:
                    results.append(plate_result)
        
        except Exception as e:
            logger.error(f"Error in detection/recognition: {e}", exc_info=True)
        
        return results
    
    def _detect_plates(self, image: np.ndarray) -> List[Detection]:
        """Detect license plates using YOLO + fallback"""
        detections = []
        
        # YOLO detection
        if self.model:
            detections.extend(self._yolo_detection(image))
        
        # Fallback contour detection if enabled
        if not detections and config.detection.use_contour_fallback:
            detections.extend(self._contour_detection(image))
        
        return detections
    
    def _yolo_detection(self, image: np.ndarray) -> List[Detection]:
        """YOLO-based detection"""
        detections = []
        
        try:
            results = self.model(image, verbose=False, device=self.device)
            
            for result in results:
                if result.boxes is None:
                    continue
                
                for box in result.boxes:
                    conf = box.conf[0].item()
                    
                    # Filter by confidence
                    if conf < config.detection.confidence_threshold:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                        method='YOLO'
                    ))
        
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
        
        return detections
    
    def _contour_detection(self, image: np.ndarray) -> List[Detection]:
        """Fallback contour-based detection"""
        detections = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < config.detection.contour_area_threshold:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if config.detection.min_aspect_ratio < aspect_ratio < config.detection.max_aspect_ratio:
                    detections.append(Detection(
                        bbox=(x, y, x + w, y + h),
                        confidence=0.5,
                        method='Contour'
                    ))
        
        except Exception as e:
            logger.error(f"Contour detection error: {e}")
        
        return detections
    
    def _process_detection(self, image: np.ndarray, detection: Detection) -> Optional[PlateResult]:
        """Extract and recognize text from a detection"""
        start_time = datetime.now()
        
        try:
            x1, y1, x2, y2 = detection.bbox
            
            # Extract ROI
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                logger.warning("Empty ROI extracted")
                return None
            
            # Preprocess
            preprocessed = self.preprocessor.preprocess_plate(roi.copy())
            
            # Correct rotation
            corrected, _ = self.preprocessor.correct_rotation(preprocessed)
            
            # Resize for OCR
            resized = self.preprocessor.resize_for_ocr(corrected)
            
            # Extract text
            text, confidence = self._extract_text(resized)
            
            # Validate and correct
            validation_result = self.validator.validate_and_correct(text, confidence)
            
            # Create OCR result
            ocr_result = OCRResult(
                text=validation_result['normalized_text'],
                confidence=confidence,
                validated=validation_result['is_valid'],
                state_code=validation_result.get('state_code'),
                registration_district=validation_result.get('registration_district'),
                series=validation_result.get('series'),
                sequential_number=validation_result.get('sequential_number'),
            )
            
            # Create plate result
            plate_result = PlateResult(
                detection=detection,
                ocr_result=ocr_result,
                processing_time=(datetime.now() - start_time).total_seconds(),
                image_roi=resized,
            )
            
            return plate_result
        
        except Exception as e:
            logger.error(f"Error processing detection: {e}", exc_info=True)
            return None
    
    def _extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using EasyOCR"""
        if self.reader is None:
            logger.warning("OCR reader not available")
            return "", 0.0
        
        try:
            results = self.reader.readtext(
                image,
                detail=1,
                allowlist=config.ocr.allowlist
            )
            
            if not results:
                return "", 0.0
            
            # Combine all text with their confidences
            texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if text.strip():
                    texts.append(text.strip())
                    confidences.append(confidence)
            
            if texts:
                combined_text = ' '.join(texts)
                avg_confidence = np.mean(confidences) if confidences else 0.0
                return combined_text, avg_confidence
            
            return "", 0.0
        
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0.0
    
    def annotate_image(self, image: np.ndarray, 
                      results: List[PlateResult]) -> np.ndarray:
        """
        Annotate image with detections and results
        
        Args:
            image: Original image
            results: Detection and recognition results
        
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for result in results:
            x1, y1, x2, y2 = result.detection.bbox
            
            # Draw bounding box
            color = (0, 255, 0) if result.ocr_result.validated else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw text
            text = result.ocr_result.text
            confidence = result.ocr_result.confidence
            label = f"{text} ({confidence:.2f})"
            
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        
        return annotated


# Example usage
if __name__ == "__main__":
    # Initialize system
    lpr_system = ImprovedIndianLPRSystem(device='cpu')
    
    # Process image
    image_path = "sample_image.jpg"
    result = lpr_system.process_image(image_path)
    
    print("\nProcessing Results:")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Processing Time: {result['processing_time']:.2f}s")
    print(f"Plates Detected: {len(result['plates'])}")
    
    for i, plate in enumerate(result['plates']):
        print(f"\nPlate {i + 1}:")
        print(f"  Text: {plate['text']}")
        print(f"  Confidence: {plate['confidence']:.2f}")
        print(f"  Validated: {plate['validated']}")
