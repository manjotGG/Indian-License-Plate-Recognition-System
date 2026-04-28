"""
License plate text validation and post-processing
Handles OCR output cleaning, correction, and validation
"""

import re
from typing import Tuple, Optional, Dict
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class IndianPlateValidator:
    """Validates and corrects Indian license plate text"""
    
    # Indian state codes
    VALID_STATE_CODES = {
        'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK',
        'JH', 'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'OD', 'PB',
        'RJ', 'SK', 'TN', 'TG', 'TR', 'UP', 'UK', 'WB', 'DL', 'CH',
        'DD', 'DN', 'LD', 'PY'
    }
    
    # Indian license plate formats
    PATTERNS = {
        'standard': r'^([A-Z]{2})\s*(\d{2})\s*([A-Z]{1,2})\s*(\d{4})$',
        'commercial': r'^([A-Z]{2})\s*(\d{2})\s*([A-Z]{1})\s*(\d{4})$',
        'old': r'^([A-Z]{2})-(\d{2})-([A-Z]{1,2})-(\d{4})$',
    }
    
    def __init__(self):
        """Initialize the validator"""
        self.char_confusion_map = {
            # Letters that look like numbers
            'O': '0', 'I': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5',
            'G': '6', 'T': '7', 'B': '8', 'g': '9',
            # Numbers that look like letters
            '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S',
            '6': 'G', '7': 'T', '8': 'B', '9': 'g',
        }
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Raw OCR output text
        
        Returns:
            Cleaned text with uppercase and removed special chars
        """
        if not text:
            return ""
        
        # Convert to uppercase
        text = text.upper().strip()
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s-]', '', text)  # Remove special chars except dash and space
        text = re.sub(r'\s+', ' ', text)      # Normalize spaces
        
        return text
    
    def validate_format(self, text: str) -> Tuple[bool, str]:
        """
        Validate if text matches Indian license plate format
        
        Args:
            text: Cleaned license plate text
        
        Returns:
            Tuple of (is_valid, normalized_text)
        """
        if not text or len(text) < 8:
            return False, text
        
        # Try each pattern
        for pattern_name, pattern in self.PATTERNS.items():
            match = re.match(pattern, text)
            if match:
                groups = match.groups()
                # Normalize: XX DD XX DDDD format
                normalized = f"{groups[0]} {groups[1]} {groups[2]} {groups[3]}"
                
                # Validate state code
                if groups[0] in self.VALID_STATE_CODES:
                    return True, normalized
        
        return False, text
    
    def correct_ocr_errors(self, text: str) -> str:
        """
        Correct common OCR errors based on position
        
        Args:
            text: License plate text with potential OCR errors
        
        Returns:
            Corrected text
        """
        text = self.clean_text(text)
        parts = text.split()
        
        if len(parts) < 3:
            return text
        
        corrected = []
        
        # Part 1: State code (2 letters) - should only have letters
        part1 = parts[0]
        corrected_part1 = ''.join(
            self.char_confusion_map.get(c, c) if c.isdigit() else c
            for c in part1
        )
        corrected.append(corrected_part1[:2])  # Ensure exactly 2 chars
        
        # Part 2: Registration district (2 digits) - should only have digits
        if len(parts) > 1:
            part2 = parts[1]
            corrected_part2 = ''.join(
                self.char_confusion_map.get(c, c) if c.isalpha() else c
                for c in part2
            )
            corrected.append(corrected_part2[:2])  # Ensure exactly 2 chars
        
        # Part 3: Series (1-2 letters) - should only have letters
        if len(parts) > 2:
            part3 = parts[2]
            corrected_part3 = ''.join(
                self.char_confusion_map.get(c, c) if c.isdigit() else c
                for c in part3
            )
            corrected.append(corrected_part3)
        
        # Part 4: Sequential number (4 digits) - should only have digits
        if len(parts) > 3:
            part4_combined = ''.join(parts[3:])  # Join remaining parts
            corrected_part4 = ''.join(
                self.char_confusion_map.get(c, c) if c.isalpha() else c
                for c in part4_combined
            )
            corrected.append(corrected_part4[-4:])  # Take last 4 chars
        
        return ' '.join(corrected)
    
    def validate_and_correct(self, text: str, confidence: float = 0.0) -> Dict:
        """
        Complete validation and correction pipeline
        
        Args:
            text: Raw OCR output
            confidence: OCR confidence score
        
        Returns:
            Dictionary with validation results
        """
        result = {
            'raw_text': text,
            'cleaned_text': '',
            'corrected_text': '',
            'normalized_text': '',
            'is_valid': False,
            'confidence': confidence,
            'message': '',
            'state_code': None,
            'registration_district': None,
            'series': None,
            'sequential_number': None,
        }
        
        # Step 1: Clean text
        cleaned = self.clean_text(text)
        result['cleaned_text'] = cleaned
        
        if not cleaned:
            result['message'] = "Empty text after cleaning"
            logger.warning(result['message'])
            return result
        
        # Step 2: Try direct format validation
        is_valid, normalized = self.validate_format(cleaned)
        if is_valid:
            result['is_valid'] = True
            result['normalized_text'] = normalized
            result['message'] = "Valid format - no corrections needed"
            self._parse_plate_components(normalized, result)
            logger.info(f"Valid plate detected: {normalized}")
            return result
        
        # Step 3: Attempt OCR error correction
        corrected = self.correct_ocr_errors(cleaned)
        result['corrected_text'] = corrected
        
        is_valid, normalized = self.validate_format(corrected)
        if is_valid:
            result['is_valid'] = True
            result['normalized_text'] = normalized
            result['message'] = "Valid after OCR correction"
            self._parse_plate_components(normalized, result)
            logger.info(f"Corrected plate detected: {normalized}")
            return result
        
        # Step 4: Apply aggressive correction if high confidence
        if confidence > 0.7:
            aggressive_corrected = self._aggressive_correction(corrected)
            result['corrected_text'] = aggressive_corrected
            
            is_valid, normalized = self.validate_format(aggressive_corrected)
            if is_valid:
                result['is_valid'] = True
                result['normalized_text'] = normalized
                result['message'] = "Valid after aggressive correction (low confidence)"
                self._parse_plate_components(normalized, result)
                return result
        
        result['message'] = "Could not validate plate format"
        logger.warning(f"Invalid plate: {text} -> {corrected}")
        
        return result
    
    def _aggressive_correction(self, text: str) -> str:
        """Apply more aggressive corrections for low-confidence OCR"""
        # Try to force valid format constraints
        text = re.sub(r'\s+', ' ', text).strip()
        parts = text.split()
        
        if len(parts) >= 3:
            # Ensure structure: 2 letters, 2 digits, 1-2 letters, 4 digits
            corrected_parts = []
            
            # Part 1: 2 letters
            p1 = ''.join(c for c in parts[0] if c.isalpha())[:2].ljust(2, 'A')
            corrected_parts.append(p1)
            
            # Part 2: 2 digits
            p2 = ''.join(c for c in parts[1] if c.isdigit())[:2].ljust(2, '0')
            corrected_parts.append(p2)
            
            # Part 3: 1-2 letters
            p3 = ''.join(c for c in parts[2] if c.isalpha())[:2]
            corrected_parts.append(p3 if p3 else 'A')
            
            # Part 4: 4 digits
            p4 = ''.join(c for c in ''.join(parts[3:]) if c.isdigit())[:4].rjust(4, '0')
            corrected_parts.append(p4)
            
            return ' '.join(corrected_parts)
        
        return text
    
    def _parse_plate_components(self, normalized_text: str, result: Dict):
        """Parse plate components from validated text"""
        parts = normalized_text.split()
        if len(parts) >= 4:
            result['state_code'] = parts[0]
            result['registration_district'] = parts[1]
            result['series'] = parts[2]
            result['sequential_number'] = parts[3]


# Example usage
if __name__ == "__main__":
    validator = IndianPlateValidator()
    
    # Test cases
    test_cases = [
        "MH 02 BC 5678",
        "MH02BC5678",
        "M H 0 2 B C 5 6 7 8",
        "DL 01 AB 1234",
        "DL0IAB1234",  # OCR error: 1 instead of 0, I instead of 1
        "KA 03 C 9876",
    ]
    
    for test_text in test_cases:
        result = validator.validate_and_correct(test_text, confidence=0.85)
        print(f"\nInput: {test_text}")
        print(f"Result: {result['normalized_text']}")
        print(f"Valid: {result['is_valid']}")
