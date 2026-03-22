"""
Quick Reference Guide for Indian ALPR System Improvements
Use this file as a cheat sheet for common tasks
"""

# ============================================================================
# 1. CONFIGURATION MANAGEMENT
# ============================================================================

from config_manager import config

# Access current settings
print(config.detection.model_path)
print(config.ocr.confidence_threshold)
print(config.preprocessing.clahe_clip_limit)

# Load from file
config.load_from_file("config.json")

# Save configuration
config.save_to_file("my_config.json")

# Modify settings programmatically
config.detection.confidence_threshold = 0.5
config.ocr.languages = ["en", "hi"]

# ============================================================================
# 2. LOGGING SETUP
# ============================================================================

from logger_util import setup_logger

# Create logger with console + file output
logger = setup_logger(__name__, log_file="app.log")

# Use in code
logger.debug("Detailed diagnostic info")
logger.info("Processing image: sample.jpg")
logger.warning("Low confidence detection: 0.35")
logger.error("Failed to load model")
logger.critical("System error - shutting down")

# ============================================================================
# 3. SIMPLE PLATE VALIDATION
# ============================================================================

from plate_validator import IndianPlateValidator

validator = IndianPlateValidator()

# Validate and correct text
raw_text = "IVH 02 BC 5678"  # Possible OCR errors
result = validator.validate_and_correct(raw_text, confidence=0.85)

print(f"Valid: {result['is_valid']}")
print(f"Corrected: {result['normalized_text']}")
print(f"State: {result['state_code']}")
print(f"District: {result['registration_district']}")

# Quick validation (returns True/False)
is_valid, normalized = validator.validate_format("MH 02 BC 5678")

# Manual cleaning
cleaned = validator.clean_text("MH    02    BC    5678")

# ============================================================================
# 4. IMAGE PREPROCESSING
# ============================================================================

from plate_preprocessor import PlatePreprocessor
import cv2

preprocessor = PlatePreprocessor()

# Load image
image = cv2.imread("plate.jpg")

# Simple preprocessing
processed = preprocessor.preprocess_plate(image)

# Preprocessing with multiple methods (returns best)
best_processed = preprocessor.preprocess_plate(image, use_all_methods=True)

# Correct rotation
corrected, angle = preprocessor.correct_rotation(processed)
print(f"Corrected {angle:.1f} degrees")

# Resize for optimal OCR
resized = preprocessor.resize_for_ocr(corrected)

# Individual enhancement techniques
enhanced_contrast = preprocessor.enhance_contrast(image, contrast_factor=1.5)
denoised = preprocessor.reduce_noise(image, method='bilateral')

# ============================================================================
# 5. MAIN ALPR SYSTEM (IMPROVED)
# ============================================================================

from improved_lpr_system import ImprovedIndianLPRSystem

# Initialize system
lpr = ImprovedIndianLPRSystem(device='cpu')  # or 'cuda' for GPU

# Process single image
result = lpr.process_image("car.jpg")

# Access results
if result['status'] == 'success':
    for plate in result['plates']:
        print(f"Text: {plate['text']}")
        print(f"Confidence: {plate['confidence']:.2%}")
        print(f"Valid: {plate['validated']}")
        print(f"State: {plate['state_code']}")
else:
    print(f"Error: {result['message']}")

# Process raw image (returns list of PlateResult objects)
import cv2
image = cv2.imread("car.jpg")
plate_results = lpr.detect_and_recognize(image)

for plate in plate_results:
    print(f"Detection: {plate.detection.bbox}")
    print(f"Text: {plate.ocr_result.text}")
    print(f"Time: {plate.processing_time:.2f}s")

# Annotate image with results
annotated = lpr.annotate_image(image, plate_results)
cv2.imwrite("output.jpg", annotated)

# ============================================================================
# 6. BATCH PROCESSING
# ============================================================================

from batch_processor import BatchProcessor

# Initialize processor
processor = BatchProcessor(device='cuda')

# Process entire directory
summary = processor.process_directory(
    input_dir='./images',
    output_dir='./results',
    recursive=True,
    save_annotated=True
)

print(f"Processed: {summary['total_images']}")
print(f"Found: {summary['total_plates']}")
print(f"Valid: {summary['valid_plates']}")

# Get statistics
stats = processor.get_statistics()
print(stats['success_rate'])
print(stats['average_time_per_image'])

# ============================================================================
# 7. STREAMLIT WEB APP
# ============================================================================

# Run the web interface:
# $ streamlit run app.py
#
# Features:
# - Single image upload/camera
# - Batch processing
# - Real-time results
# - Download JSON results

# ============================================================================
# 8. COMMON WORKFLOWS
# ============================================================================

# WORKFLOW 1: Simple one-off detection
def detect_single_plate(image_path):
    lpr = ImprovedIndianLPRSystem()
    result = lpr.process_image(image_path)
    return result['plates'][0] if result['plates'] else None

# WORKFLOW 2: Batch process with progress
def process_batch_with_progress(image_dir):
    processor = BatchProcessor()
    
    from pathlib import Path
    images = list(Path(image_dir).glob('*.jpg'))
    
    results = []
    for idx, img_path in enumerate(images):
        print(f"Processing {idx + 1}/{len(images)}")
        result = processor.lpr_system.process_image(str(img_path))
        results.append(result)
    
    return results

# WORKFLOW 3: Advanced preprocessing + OCR
def high_quality_detection(image_path):
    import cv2
    
    # Setup
    preprocessor = PlatePreprocessor()
    validator = IndianPlateValidator()
    lpr = ImprovedIndianLPRSystem()
    
    # Load and preprocess
    image = cv2.imread(image_path)
    plates = lpr.detect_and_recognize(image)
    
    # Enhanced processing
    enhanced_results = []
    for plate in plates:
        # Use best preprocessing method
        roi = image[plate.detection.bbox[1]:plate.detection.bbox[3],
                    plate.detection.bbox[0]:plate.detection.bbox[2]]
        
        processed = preprocessor.preprocess_plate(roi, use_all_methods=True)
        corrected, _ = preprocessor.correct_rotation(processed)
        
        # Validate
        validation = validator.validate_and_correct(
            plate.ocr_result.text,
            confidence=plate.ocr_result.confidence
        )
        
        enhanced_results.append(validation)
    
    return enhanced_results

# WORKFLOW 4: Video processing (frame by frame)
def process_video(video_path):
    import cv2
    
    lpr = ImprovedIndianLPRSystem()
    cap = cv2.VideoCapture(video_path)
    
    all_detections = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame for speed
        if frame_count % 5 == 0:
            plates = lpr.detect_and_recognize(frame)
            for plate in plates:
                all_detections.append({
                    'frame': frame_count,
                    'text': plate.ocr_result.text,
                    'confidence': plate.ocr_result.confidence
                })
        
        frame_count += 1
    
    cap.release()
    return all_detections

# ============================================================================
# 9. ERROR HANDLING
# ============================================================================

from logger_util import setup_logger

logger = setup_logger(__name__)

try:
    lpr = ImprovedIndianLPRSystem()
    result = lpr.process_image("image.jpg")
    
    if result['status'] != 'success':
        logger.error(f"Processing failed: {result['message']}")
    else:
        logger.info(f"Found {len(result['plates'])} plates")

except FileNotFoundError:
    logger.error("Image file not found")
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)

# ============================================================================
# 10. PERFORMANCE TIPS
# ============================================================================

# TIP 1: Reuse model instance
lpr = ImprovedIndianLPRSystem()
for image_path in ["img1.jpg", "img2.jpg", "img3.jpg"]:
    result = lpr.process_image(image_path)  # Reuses loaded model

# TIP 2: Use GPU for batch processing
processor = BatchProcessor(device='cuda')
summary = processor.process_directory('./images')

# TIP 3: Cache configuration
config.load_from_file("config.json")
lpr = ImprovedIndianLPRSystem()  # Uses cached config

# TIP 4: Preprocess images efficiently
from plate_preprocessor import PlatePreprocessor
preprocessor = PlatePreprocessor()

# Process multiple images with same preprocessor
for image_path in image_list:
    image = cv2.imread(image_path)
    processed = preprocessor.preprocess_plate(image)
    # Send to OCR

# ============================================================================
# 11. DEBUGGING TIPS
# ============================================================================

# TIP 1: Enable detailed logging
logger = setup_logger(__name__, log_file="debug.log")

# TIP 2: Save intermediate results
def debug_pipeline(image_path):
    import cv2
    from plate_preprocessor import PlatePreprocessor
    
    preprocessor = PlatePreprocessor()
    matrix = cv2.imread(image_path)
    
    processed = preprocessor.preprocess_plate(image)
    cv2.imwrite("debug_01_processed.jpg", processed)
    
    corrected, _ = preprocessor.correct_rotation(processed)
    cv2.imwrite("debug_02_corrected.jpg", corrected)
    
    resized = preprocessor.resize_for_ocr(corrected)
    cv2.imwrite("debug_03_resized.jpg", resized)

# TIP 3: Inspect model predictions
def inspect_detections(image_path):
    lpr = ImprovedIndianLPRSystem()
    image = cv2.imread(image_path)
    
    detections = lpr._detect_plates(image)
    for i, det in enumerate(detections):
        print(f"Detection {i}: bbox={det.bbox}, conf={det.confidence:.2%}, method={det.method}")

# ============================================================================
# 12. USEFUL SNIPPETS
# ============================================================================

# Save all detections to CSV
import pandas as pd

def save_results_to_csv(results, output_file):
    data = []
    for result in results:
        for plate in result.get('plates', []):
            data.append({
                'image': result['image_path'],
                'text': plate['text'],
                'confidence': plate['confidence'],
                'validated': plate['validated'],
                'state': plate['state_code']
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

# Find all valid plates in results
def filter_valid_plates(results):
    valid_plates = []
    for result in results:
        for plate in result.get('plates', []):
            if plate['validated']:
                valid_plates.append(plate)
    return valid_plates

# Get statistics from results
def get_result_statistics(results):
    total_images = len(results)
    total_plates = sum(len(r.get('plates', [])) for r in results)
    valid_plates = sum(1 for r in results for p in r.get('plates', []) if p['validated'])
    
    return {
        'total_images': total_images,
        'total_plates': total_plates,
        'valid_plates': valid_plates,
        'success_rate': f"{(valid_plates / total_plates * 100):.1f}%" if total_plates > 0 else "N/A"
    }

# ============================================================================
# END OF QUICK REFERENCE
# ============================================================================
