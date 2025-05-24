"""
Streamlit web application for Wavelet-based Style Transfer.

This app provides an interactive interface to apply WCT2 style transfer
to images with optional segmentation maps.
"""
import os
import torch
import numpy as np
from PIL import Image
import streamlit as st
from torchvision.utils import save_image

from model import WaveEncoder, WaveDecoder
from utils.core import feature_wct
from utils.io import Timer, open_image, load_segment, compute_label_info
from transfer import WCT2, is_image_file

# Set page configuration
st.set_page_config(
    page_title="WCT2 Style Transfer",
    page_icon="🎨",
    layout="wide"
)

# Create necessary directories
@st.cache_resource
def create_directories():
    """Create necessary directories for the application."""
    dirs = [
        "./uploads/content",
        "./uploads/style",
        "./uploads/content_segment",
        "./uploads/style_segment",
        "./results"
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

create_directories()

# Sidebar for configuration options
st.sidebar.title("Style Transfer Settings")

# Device selection
device_option = st.sidebar.radio(
    "Select Device",
    ["CPU", "CUDA (if available)"],
    index=1 if torch.cuda.is_available() else 0
)
device = torch.device('cuda:0' if device_option == "CUDA (if available)" and torch.cuda.is_available() else 'cpu')

# Transfer location options
st.sidebar.subheader("Transfer Locations")
transfer_at_encoder = st.sidebar.checkbox("Encoder", value=True)
transfer_at_decoder = st.sidebar.checkbox("Decoder", value=False)
transfer_at_skip = st.sidebar.checkbox("Skip Connections", value=False)

# Get selected transfer locations
def get_transfer_locations():
    locations = []
    if transfer_at_encoder:
        locations.append('encoder')
    if transfer_at_decoder:
        locations.append('decoder')
    if transfer_at_skip:
        locations.append('skip')
    return locations if locations else ['encoder']  # Default to encoder if nothing selected

# Other configuration options
option_unpool = st.sidebar.selectbox("Unpooling Method", ["cat5", "sum"], index=0)
image_size = st.sidebar.slider("Image Size", min_value=256, max_value=1024, value=512, step=64)
alpha = st.sidebar.slider("Style Weight (Alpha)", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
verbose = st.sidebar.checkbox("Verbose Output", value=False)

# Main app interface
st.title("WCT2: Wavelet-based Style Transfer")
st.markdown("""
This application allows you to apply photorealistic style transfer using the WCT2 algorithm, 
which leverages wavelet transforms to create high-quality stylized images.

Upload your content and style images (and optional segmentation maps) to get started.
""")

# File upload section
st.header("Upload Images")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Content Image")
    content_file = st.file_uploader("Upload content image", type=['png', 'jpg', 'jpeg'])
    if content_file:
        content_image = Image.open(content_file)
        st.image(content_image, caption="Content Image", use_column_width=True)
        content_path = os.path.join("./uploads/content", content_file.name)
        content_image.save(content_path)

    st.subheader("Content Segmentation (Optional)")
    content_segment_file = st.file_uploader("Upload content segmentation", type=['png', 'jpg', 'jpeg'])
    if content_segment_file:
        content_segment_image = Image.open(content_segment_file)
        st.image(content_segment_image, caption="Content Segmentation", use_column_width=True)
        content_segment_path = os.path.join("./uploads/content_segment", content_segment_file.name)
        content_segment_image.save(content_segment_path)

with col2:
    st.subheader("Style Image")
    style_file = st.file_uploader("Upload style image", type=['png', 'jpg', 'jpeg'])
    if style_file:
        style_image = Image.open(style_file)
        st.image(style_image, caption="Style Image", use_column_width=True)
        style_path = os.path.join("./uploads/style", style_file.name)
        style_image.save(style_path)

    st.subheader("Style Segmentation (Optional)")
    style_segment_file = st.file_uploader("Upload style segmentation", type=['png', 'jpg', 'jpeg'])
    if style_segment_file:
        style_segment_image = Image.open(style_segment_file)
        st.image(style_segment_image, caption="Style Segmentation", use_column_width=True)
        style_segment_path = os.path.join("./uploads/style_segment", style_segment_file.name)
        style_segment_image.save(style_segment_path)

# Perform style transfer
def perform_style_transfer(content_path, style_path, content_segment_path=None, style_segment_path=None):
    """
    Apply WCT2 style transfer and return the result.
    
    Args:
        content_path: Path to content image
        style_path: Path to style image
        content_segment_path: Optional path to content segmentation
        style_segment_path: Optional path to style segmentation
        
    Returns:
        Path to the result image
    """
    # Load images
    content_tensor = open_image(content_path, image_size).to(device)
    style_tensor = open_image(style_path, image_size).to(device)
    
    # Load segmentation maps if available
    content_segment = load_segment(content_segment_path, image_size) if content_segment_path else np.asarray([])
    style_segment = load_segment(style_segment_path, image_size) if style_segment_path else np.asarray([])
    
    # Set up transfer locations
    transfer_locations = get_transfer_locations()
    
    # Initialize WCT2 model
    wct_model = WCT2(
        transfer_at=transfer_locations,
        option_unpool=option_unpool,
        device=device,
        verbose=verbose
    )
    
    # Perform style transfer
    with torch.no_grad():
        with Timer('Style transfer completed in: {}', verbose):
            result = wct_model.transfer(
                content_tensor, 
                style_tensor,
                content_segment, 
                style_segment,
                alpha=alpha
            )
    
    # Save and return result
    result_filename = f"result_{'-'.join(transfer_locations)}_{option_unpool}.png"
    result_path = os.path.join("./results", result_filename)
    save_image(result.clamp_(0, 1), result_path, padding=0)
    
    return result_path

# Transfer button
if st.button("Apply Style Transfer"):
    if content_file and style_file:
        with st.spinner("Applying style transfer... This may take a minute."):
            try:
                content_path = os.path.join("./uploads/content", content_file.name)
                style_path = os.path.join("./uploads/style", style_file.name)
                
                content_segment_path = None
                if content_segment_file:
                    content_segment_path = os.path.join("./uploads/content_segment", content_segment_file.name)
                
                style_segment_path = None
                if style_segment_file:
                    style_segment_path = os.path.join("./uploads/style_segment", style_segment_file.name)
                
                result_path = perform_style_transfer(
                    content_path, 
                    style_path, 
                    content_segment_path, 
                    style_segment_path
                )
                
                st.success("Style transfer completed!")
                
                # Display result
                st.header("Result")
                st.image(result_path, caption="Stylized Image", use_column_width=True)
                
                # Download link
                with open(result_path, "rb") as file:
                    btn = st.download_button(
                        label="Download stylized image",
                        data=file,
                        file_name=os.path.basename(result_path),
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"An error occurred during style transfer: {str(e)}")
    else:
        st.warning("Please upload both content and style images.")

# Information section
st.markdown("""
## About WCT2 Style Transfer

WCT2 (Wavelet Corrective Transfer) is a photorealistic style transfer algorithm that uses wavelet transforms
to maintain the structure of the content image while transferring the style effectively.

### Transfer Locations:

- **Encoder**: Applies style transfer at the encoder features (creates stronger style effects)
- **Decoder**: Applies style transfer at the decoder features
- **Skip Connections**: Applies style transfer at the wavelet components

### Tips:

- For photorealistic results, use segmentation maps to control style transfer in specific regions
- Adjust the style weight (alpha) to control the strength of style transfer
- Try different combinations of transfer locations for varied effects
""")
