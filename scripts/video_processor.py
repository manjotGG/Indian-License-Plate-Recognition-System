"""
Video stream processing for ALPR system
Supports real-time video processing from files or streams
"""

import cv2
import time
from pathlib import Path
from typing import Callable, Optional
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.core.lpr_system import ImprovedIndianLPRSystem
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VideoProcessor:
    """Process video streams for license plate recognition"""
    
    def __init__(self, lpr_system: ImprovedIndianLPRSystem, 
                 fps: int = 1, confidence_threshold: float = 0.5):
        """
        Initialize video processor
        
        Args:
            lpr_system: ALPR system instance
            fps: Frames per second to process
            confidence_threshold: Minimum confidence for saving results
        """
        self.lpr_system = lpr_system
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        self.frame_interval = 1.0 / fps if fps > 0 else 0
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None,
                          callback: Optional[Callable] = None) -> Dict:
        """
        Process video file
        
        Args:
            video_path: Path to video file
            output_path: Path to save annotated video
            callback: Callback function for each frame result
        
        Returns:
            Processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'plates_detected': 0,
            'processing_time': 0,
            'unique_plates': set()
        }
        
        start_time = time.time()
        last_process_time = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time() - start_time
                
                # Process frame at specified FPS
                if current_time - last_process_time >= self.frame_interval:
                    frame_start = time.time()
                    
                    # Process frame
                    result = self.lpr_system.process_image_from_array(frame, f"frame_{stats['processed_frames']}.jpg")
                    
                    frame_time = time.time() - frame_start
                    stats['processing_time'] += frame_time
                    stats['processed_frames'] += 1
                    
                    # Count detections
                    for plate in result.get('plates', []):
                        if plate.get('confidence', 0) >= self.confidence_threshold:
                            stats['plates_detected'] += 1
                            stats['unique_plates'].add(plate.get('plate_text', ''))
                    
                    # Annotate frame
                    if out:
                        annotated = self._annotate_frame(frame, result)
                        out.write(annotated)
                    
                    # Callback
                    if callback:
                        callback(result, stats)
                    
                    last_process_time = current_time
                    
                    # Progress logging
                    if stats['processed_frames'] % 100 == 0:
                        logger.info(f"Processed {stats['processed_frames']}/{total_frames} frames")
        
        finally:
            cap.release()
            if out:
                out.release()
        
        stats['unique_plates'] = list(stats['unique_plates'])
        stats['avg_time_per_frame'] = stats['processing_time'] / max(stats['processed_frames'], 1)
        
        logger.info(f"Video processing complete: {stats}")
        return stats
    
    def _annotate_frame(self, frame, result: Dict):
        """Annotate frame with detection results"""
        annotated = frame.copy()
        
        for plate in result.get('plates', []):
            bbox = plate.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                text = plate.get('plate_text', '')
                confidence = plate.get('confidence', 0)
                label = f"{text} ({confidence:.2f})"
                
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated


# Example usage
if __name__ == "__main__":
    # Initialize ALPR system
    lpr_system = ImprovedIndianLPRSystem(device='cpu', enable_database=True)
    
    # Initialize video processor
    video_processor = VideoProcessor(lpr_system, fps=1)  # Process 1 frame per second
    
    # Process video
    stats = video_processor.process_video_file(
        "sample_video.mp4",
        output_path="annotated_video.mp4"
    )
    
    print(f"Processing complete: {stats}")