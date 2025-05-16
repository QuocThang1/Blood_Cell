# Medical Image Segmentation

Group 4

- 22110007 Nguyen Nhat An
- 22110028 Nguyen Mai Huy Hoang
- 22110070 Dinh To Quoc Thang
- 22110076 Tran Trung Tin

## Blood Cell Analysis Platform

A sophisticated application for automated blood cell segmentation and classification using image processing and deep learning techniques.

## Overview

This project implements advanced image processing and pattern recognition techniques to segment and classify blood cells from microscopic images. It provides an interactive, user-friendly interface for medical professionals to analyze blood samples efficiently and accurately.

## Features

- **Automated Segmentation**: Utilizes a U-Net deep learning architecture to precisely segment blood cells from microscopic images
- **Cell Classification**: Identifies and classifies white blood cells into five main types (Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil)
- **Interactive Visualization**: Multiple view modes including original, segmentation mask, and overlay
- **Detailed Analysis**: Histogram analysis, cell feature extraction, and confidence metrics

## Installation

### Requirements

- Python 3.7+
- PyQt5
- PyTorch
- OpenCV
- scikit-image
- Matplotlib
- NumPy

### Setup

1. Clone this repository:

   ```
   git clone https://github.com/QuocThang1/Blood_Cell.git
   cd Blood_Cell
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Download model files:

   - Download the pre-trained model files from [Google Drive](https://drive.google.com/drive/folders/1ZiXHUu2obyGsv7l_4m9StOseHzgjkh4z?usp=sharing)
   - **unet_best-8-5-16-26.pth**: Segmentation model
   - **classifier.pkl**: Cell classification model

4. Create a models folder and place the downloaded files inside:
   ```
   mkdir -p models
   # Copy the downloaded .pth and .pkl files to the models folder
   ```

## Usage

Run the application:

```
python gui.py
```

### Basic Workflow

1. **Load Image**: Click "Load Image" to open a blood cell microscope image
2. **Segment Image**: Process the image to isolate individual cells
3. **Classify Cells**: Identify the type of white blood cells present
4. **Analyze Results**: View segmentation results and classification details

## Architecture

The application uses a U-Net convolutional neural network for segmentation and a pretrained classifier for white blood cell type identification. The image processing pipeline includes:

1. Image preprocessing and normalization
2. Deep learning-based segmentation
3. Feature extraction from segmented regions
4. Classification based on extracted features
5. Result visualization and reporting

## Acknowledgments

- PyTorch for deep learning frameworks
- PyQt5 for the graphical user interface
- OpenCV and scikit-image for image processing capabilities
