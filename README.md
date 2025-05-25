# Photorealistic Style Transfer - WCT2

A PyTorch implementation of the paper ["A Closed-form Solution to Photorealistic Image Stylization"](https://arxiv.org/pdf/1802.06474), featuring a wavelet-based approach to neural style transfer.

## Overview

This project implements a two-step process for photorealistic style transfer:
1. **Stylization Step**: Whitening and Coloring Transforms (WCT)
2. **Smoothing Step**: Manifold ranking algorithm to preserve spatial consistency

## Key Features

- End-to-end model that stylizes 1024×1024 images in 4.7 seconds
- Maintains photorealistic textures and structures
- Handles transformations like day-to-night or season changes
- Stable video stylization across frames

## Results

The model demonstrates effective style transfer while preserving the photorealistic nature of the content images:

![Style Transfer Results](docs/images/pyramids1.jpg) + ![Style Image](docs/images/pyramids2.jpg) → ![Output](docs/images/pyramids_output.jpg)

## Technical Details

- **Framework**: PyTorch
- **Techniques**: 
  - WCT (Whitening and Coloring Transform)
  - Wavelet transforms for multi-level feature extraction
- **Performance**: Real-time stylization of high-resolution images

## Project Structure

```
.
├── docs/               # GitHub Pages website
├── examples/          # Example images
│   ├── content/      # Content images
│   └── style/        # Style images
├── model.py          # Core model implementation
├── transfer.py       # Style transfer functionality
├── app.py           # Web interface
└── requirements.txt  # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khaled-gad/vision_final_project.git
cd vision_final_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the web interface:
```bash
streamlit run app.py
```

## Resources

- [Paper: A Closed-form Solution to Photorealistic Image Stylization](https://arxiv.org/pdf/1802.06474)

## Contact

For questions or collaborations, feel free to reach out:
- Khaled Gad: [khaled.gad@ejust.edu.eg](mailto:khaled.gad@ejust.edu.eg)
- Steven Yakoub: [steven.yakoub@ejust.edu.eg](mailto:steven.yakoub@ejust.edu.eg)
- Anas Awadallah: [anas.awadallah@ejust.edu.eg](mailto:anas.awadallah@ejust.edu.eg)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 