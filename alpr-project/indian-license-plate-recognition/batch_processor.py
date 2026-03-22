"""
Batch processing script for efficient image processing
Demonstrates best practices for handling multiple images
"""

import cv2
import pandas as pd
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import time
import json

from improved_lpr_system import ImprovedIndianLPRSystem
from logger_util import setup_logger

logger = setup_logger(__name__, log_file="batch_processing.log")


class BatchProcessor:
    """Batch processing for multiple images"""
    
    def __init__(self, device: str = 'cpu', use_gpu: bool = False):
        """
        Initialize batch processor
        
        Args:
            device: 'cpu' or 'cuda'
            use_gpu: Force GPU usage
        """
        self.device = device
        self.lpr_system = ImprovedIndianLPRSystem(device=device, use_gpu=use_gpu)
        self.results = []
    
    def process_directory(self, input_dir: str, output_dir: str = "results",
                         recursive: bool = True, save_annotated: bool = True) -> Dict:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory to save results
            recursive: Process subdirectories
            save_annotated: Save annotated images
        
        Returns:
            Summary of processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        if recursive:
            image_files = list(input_path.glob('**/*.jp*g')) + list(input_path.glob('**/*.png'))
        else:
            image_files = list(input_path.glob('*.jp*g')) + list(input_path.glob('*.png'))
        
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return {'total': 0, 'processed': 0, 'plates_found': 0}
        
        logger.info(f"Found {len(image_files)} images to process")
        
        total_plates = 0
        valid_plates = 0
        start_time = time.time()
        
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            print(f"\rProcessing {idx}/{len(image_files)}: {image_path.name}", end='')
            
            try:
                # Process single image
                result = self.lpr_system.process_image(str(image_path))
                
                # Count plates
                plates_found = len(result.get('plates', []))
                valid = sum(1 for p in result.get('plates', []) if p['validated'])
                
                total_plates += plates_found
                valid_plates += valid
                
                # Store result
                self.results.append({
                    'image': image_path.name,
                    'status': result['status'],
                    'plates_found': plates_found,
                    'valid_plates': valid,
                    'detections': result.get('plates', []),
                    'processing_time': result['processing_time']
                })
                
                # Save annotated image if requested
                if save_annotated and result.get('status') == 'success' and result.get('plates'):
                    self._save_annotated_image(image_path, result, output_path)
            
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                self.results.append({
                    'image': image_path.name,
                    'status': 'error',
                    'error': str(e),
                    'plates_found': 0
                })
        
        print()  # New line after progress
        
        # Calculate statistics
        processing_time = time.time() - start_time
        
        summary = {
            'total_images': len(image_files),
            'processed': len([r for r in self.results if r['status'] == 'success']),
            'failed': len([r for r in self.results if r['status'] == 'error']),
            'total_plates': total_plates,
            'valid_plates': valid_plates,
            'invalid_plates': total_plates - valid_plates,
            'total_time': processing_time,
            'avg_time_per_image': processing_time / len(image_files),
        }
        
        # Save results
        self._save_results(output_path, summary)
        
        logger.info(f"Batch processing complete: {summary}")
        
        return summary
    
    def _save_annotated_image(self, image_path: Path, result: Dict, output_dir: Path):
        """Save annotated image with detections"""
        try:
            image = cv2.imread(str(image_path))
            annotated = self.lpr_system.annotate_image(image, [])  # Add results if needed
            
            output_file = output_dir / f"annotated_{image_path.name}"
            cv2.imwrite(str(output_file), annotated)
        
        except Exception as e:
            logger.warning(f"Could not save annotated image: {e}")
    
    def _save_results(self, output_dir: Path, summary: Dict):
        """Save processing results"""
        # Save summary
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_file}")
        
        # Save detailed results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Detailed results saved to {results_file}")
        
        # Save as CSV for easy viewing
        csv_file = output_dir / "results.csv"
        df_data = []
        for result in self.results:
            plates_text = ', '.join([d.get('text', '') for d in result.get('detections', [])])
            df_data.append({
                'Image': result['image'],
                'Status': result['status'],
                'Plates Found': result.get('plates_found', 0),
                'Valid Plates': result.get('valid_plates', 0),
                'Detected Text': plates_text,
                'Processing Time (s)': result.get('processing_time', 0)
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False)
        logger.info(f"CSV results saved to {csv_file}")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        if not self.results:
            return {}
        
        total_images = len(self.results)
        successful = len([r for r in self.results if r['status'] == 'success'])
        failed = len([r for r in self.results if r['status'] == 'error'])
        
        total_plates = sum(r.get('plates_found', 0) for r in self.results)
        valid_plates = sum(r.get('valid_plates', 0) for r in self.results)
        
        total_time = sum(r.get('processing_time', 0) for r in self.results)
        
        return {
            'total_images': total_images,
            'successful': successful,
            'failed': failed,
            'success_rate': f"{(successful / total_images * 100):.1f}%" if total_images > 0 else "0%",
            'total_plates': total_plates,
            'valid_plates': valid_plates,
            'invalid_plates': total_plates - valid_plates,
            'average_confidence': 0.0,  # Can be calculated from detections
            'total_processing_time': f"{total_time:.2f}s",
            'average_time_per_image': f"{(total_time / total_images):.2f}s" if total_images > 0 else "0s",
        }


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process images for license plate recognition")
    parser.add_argument('--input', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output', type=str, default='results', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Compute device')
    parser.add_argument('--no-annotated', action='store_true', help='Do not save annotated images')
    parser.add_argument('--no-recursive', action='store_true', help='Do not process subdirectories')
    
    args = parser.parse_args()
    
    # Initialize processor
    logger.info("Initializing batch processor...")
    processor = BatchProcessor(device=args.device)
    
    # Process directory
    logger.info(f"Processing images from {args.input}")
    summary = processor.process_directory(
        input_dir=args.input,
        output_dir=args.output,
        recursive=not args.no_recursive,
        save_annotated=not args.no_annotated
    )
    
    # Print statistics
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("="*60)


if __name__ == "__main__":
    main()
