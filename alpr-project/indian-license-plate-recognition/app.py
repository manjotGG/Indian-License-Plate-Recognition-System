"""
Streamlit UI for Indian License Plate Recognition System
Simple, user-friendly web interface for the ALPR system
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
from io import BytesIO
from PIL import Image
import json
from datetime import datetime

# Import ALPR system
from improved_lpr_system import ImprovedIndianLPRSystem

# Page configuration
st.set_page_config(
    page_title="Indian License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
        .main {
            max-width: 1200px;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #28a745;
            padding: 1rem;
            border-radius: 0.25rem;
        }
        .error-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 1rem;
            border-radius: 0.25rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'lpr_system' not in st.session_state:
    with st.spinner('Initializing ALPR System...'):
        device = 'cuda' if st.session_state.get('use_gpu', False) else 'cpu'
        st.session_state.lpr_system = ImprovedIndianLPRSystem(device=device)

if 'processing_results' not in st.session_state:
    st.session_state.processing_results = None


def main():
    """Main Streamlit application"""
    
    # Title and description
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("# 🚗 ALPR System")
    with col2:
        st.markdown("**Automatic Indian License Plate Recognition** using YOLOv8 & EasyOCR")
    
    st.divider()
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    mode = st.sidebar.radio(
        "Select Mode",
        ["Single Image", "Batch Processing", "Demo"],
        help="Choose how you want to process images"
    )
    
    st.sidebar.divider()
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        detection_conf = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Minimum confidence for plate detection"
        )
        
        ocr_conf = st.slider(
            "OCR Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.1,
            help="Minimum confidence for text recognition"
        )
        
        use_fallback = st.checkbox(
            "Use Contour Fallback Detection",
            value=True,
            help="Enable fallback when YOLO detection fails"
        )
    
    st.sidebar.divider()
    
    # Main content
    if mode == "Single Image":
        single_image_mode(detection_conf, ocr_conf)
    
    elif mode == "Batch Processing":
        batch_processing_mode(detection_conf, ocr_conf)
    
    elif mode == "Demo":
        demo_mode()


def single_image_mode(detection_conf: float, ocr_conf: float):
    """Single image processing mode"""
    st.subheader("📷 Single Image Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### Or Use Camera")
        camera_image = st.camera_input("Capture image")
    
    # Process uploaded or camera image
    image_file = uploaded_file or camera_image
    
    if image_file is not None:
        # Display uploaded image
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("🔍 Detect License Plates", use_container_width=True):
            with st.spinner("Processing image..."):
                # Convert PIL to OpenCV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Process
                results = st.session_state.lpr_system.detect_and_recognize(cv_image)
                
                # Display results
                if results:
                    st.success(f"✅ Detected {len(results)} plate(s)")
                    
                    # Create tabs for each detection
                    tabs = st.tabs([f"Plate {i+1}" for i in range(len(results))])
                    
                    for i, (tab, result) in enumerate(zip(tabs, results)):
                        with tab:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Detected Text**")
                                st.markdown(f"### {result.ocr_result.text}")
                                
                                st.markdown("**Metrics**")
                                metrics_col1, metrics_col2 = st.columns(2)
                                with metrics_col1:
                                    st.metric("OCR Confidence", f"{result.ocr_result.confidence:.2%}")
                                with metrics_col2:
                                    st.metric("Detection Confidence", f"{result.detection.confidence:.2%}")
                                
                                st.markdown("**Text Validation**")
                                status = "✓ Valid" if result.ocr_result.validated else "✗ Invalid"
                                st.write(status)
                            
                            with col2:
                                # Extract and display plate ROI
                                if result.image_roi is not None:
                                    roi_rgb = cv2.cvtColor(result.image_roi, cv2.COLOR_BGR2RGB)
                                    st.image(roi_rgb, caption="Plate Region", use_column_width=True)
                            
                            # Detailed information
                            if result.ocr_result.state_code:
                                st.markdown("**Plate Components**")
                                comp_col1, comp_col2 = st.columns(2)
                                with comp_col1:
                                    st.code(f"State: {result.ocr_result.state_code}", language="")
                                with comp_col2:
                                    st.code(f"District: {result.ocr_result.registration_district}", language="")
                
                else:
                    st.warning("❌ No license plates detected in the image")
                    st.info("Try uploading a clearer image of a vehicle with visible license plate")


def batch_processing_mode(detection_conf: float, ocr_conf: float):
    """Batch image processing mode"""
    st.subheader("📁 Batch Processing")
    
    # File uploader for multiple images
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📊 {len(uploaded_files)} images selected")
        
        if st.button("🔄 Process All Images", use_container_width=True):
            # Create progress bar
            progress_bar = st.progress(0)
            results_container = st.container()
            
            all_results = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress, text=f"Processing {idx + 1}/{len(uploaded_files)}")
                
                # Process image
                image = Image.open(uploaded_file)
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                results = st.session_state.lpr_system.detect_and_recognize(cv_image)
                
                all_results.append({
                    'filename': uploaded_file.name,
                    'plates_found': len(results),
                    'detections': [
                        {
                            'text': r.ocr_result.text,
                            'confidence': r.ocr_result.confidence,
                            'validated': r.ocr_result.validated
                        }
                        for r in results
                    ]
                })
            
            # Display results
            progress_bar.empty()
            
            st.markdown("### 📈 Batch Processing Results")
            
            total_plates = sum(r['plates_found'] for r in all_results)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Images Processed", len(all_results))
            with col2:
                st.metric("Total Plates Found", total_plates)
            with col3:
                valid_plates = sum(
                    1 for r in all_results 
                    for d in r['detections'] 
                    if d['validated']
                )
                st.metric("Valid Plates", valid_plates)
            
            # Display detailed results
            for result in all_results:
                with st.expander(f"📄 {result['filename']} ({result['plates_found']} plate(s))"):
                    if result['detections']:
                        for i, detection in enumerate(result['detections']):
                            st.write(f"**Plate {i+1}:** {detection['text']}")
                            st.write(f"Confidence: {detection['confidence']:.2%}")
                            status = "✓ Valid" if detection['validated'] else "✗ Invalid"
                            st.write(f"Status: {status}")
                    else:
                        st.write("No plates detected")
            
            # Download results as JSON
            json_results = json.dumps(all_results, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_results,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )


def demo_mode():
    """Demo mode with sample images and instructions"""
    st.subheader("🎬 Demo & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📝 How to Use
        
        1. **Single Image Mode:**
           - Upload or capture an image
           - Click "Detect License Plates"
           - View results in interactive tabs
        
        2. **Batch Processing:**
           - Upload multiple images
           - Process all at once
           - Download results as JSON
        
        3. **Supported Formats:**
           - JPG, JPEG, PNG, BMP
           - Clear images work best
           - Good lighting improves accuracy
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Best Practices
        
        - **Image Quality:** Clear, well-lit images are essential
        - **Angle:** Plates should be fairly straight (not heavily tilted)
        - **Resolution:** Higher resolution images work better
        - **Background:** Minimal background interference helps
        
        ### 📊 Supported Plate Formats
        
        - **Standard:** XX DD XX DDDD
        - **Old Format:** XX-DD-XX-DDDD
        - **Commercial:** XX DD X DDDD
        
        ### ✨ Features
        
        - Real-time license plate detection
        - Automatic text recognition
        - Format validation
        - Confidence scoring
        """)
    
    st.divider()
    
    # System information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Detection Model", "YOLOv8n")
    
    with col2:
        st.metric("OCR Engine", "EasyOCR")
    
    with col3:
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.metric("Compute Device", device)


if __name__ == "__main__":
    # Import torch for device check
    try:
        import torch
    except ImportError:
        st.error("PyTorch not installed")
    
    main()
