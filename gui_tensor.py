import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QFileDialog, QComboBox, QSlider, QGroupBox,
                            QStatusBar, QSplitter, QProgressBar, QFrame, QToolButton)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor, QLinearGradient, QBrush
from PyQt5.QtCore import Qt, QSize, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage import morphology, filters, exposure
from skimage.feature import canny
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
from tensorflow.keras.models import load_model
import joblib # Add this import

try:
    model = load_model("d:/Study/University/Semester2-Year3/Digital Image Processing/Final Project/main/Blood_Cell/models/unet_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

class MedicalImageSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set application style and colors
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f4f8;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #6c8ebf;
                border-radius: 8px;
                margin-top: 1ex;
                background-color: #e9eff8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #2c3e50;
                background-color: #d4e2f4;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
            QLabel {
                color: #2c3e50;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)
        
        self.initUI()
        
        # Initialize variables
        self.original_image = None
        self.segmented_mask = None
        self.segmented_image = None
        self.white_cell_roi = None
        self.classifier = self.load_classifier()
        
        # Create a status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
    def load_classifier(self):
        """
        In real application, load a pre-trained classifier for white cell classification
        Here we mock up a simple classifier
        """
        # This is a placeholder - in a real application, you would load a trained model
        # For example: return pickle.load(open('white_cell_classifier.pkl', 'rb'))
        
        # Mock classifier - in real app, this would be replaced with a trained model
        class MockClassifier:
            def predict(self, features):
                # For demo purposes, return a random class
                cell_types = ['Neutrophil', 'Lymphocyte', 'Monocyte', 'Eosinophil', 'Basophil']
                return [np.random.choice(cell_types)]
        
        return MockClassifier()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle("Medical Image Segmentation and Classification")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application icon if you have one
        # self.setWindowIcon(QIcon('icon.png'))
        
        # Set application font
        font = QFont("Segoe UI", 10)
        QApplication.setFont(font)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel container with styled border
        left_container = QFrame()
        left_container.setFrameShape(QFrame.StyledPanel)
        left_container.setAutoFillBackground(True)
        palette = left_container.palette()
        palette.setColor(QPalette.Window, QColor("#e6eef5"))
        left_container.setPalette(palette)
        
        # Left panel - controls
        left_panel = QVBoxLayout(left_container)
        left_panel.setContentsMargins(10, 10, 10, 10)
        left_panel.setSpacing(15)
        
        # App title and logo
        title_layout = QHBoxLayout()
        app_logo = QLabel()
        app_logo.setPixmap(self.create_icon_pixmap("microscope", QSize(40, 40)))
        title_layout.addWidget(app_logo)
        
        app_title = QLabel("Medical Image Analysis")
        app_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        title_layout.addWidget(app_title, 1)
        left_panel.addLayout(title_layout)
        
        # Horizontal line
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        h_line.setStyleSheet("background-color: #6c8ebf;")
        left_panel.addWidget(h_line)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        # Load image button with icon
        self.load_button = QPushButton("  Load Image")
        self.load_button.setIcon(self.create_icon("folder-open"))
        self.load_button.setIconSize(QSize(20, 20))
        self.load_button.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_button)
        
        # Save results button with icon
        self.save_button = QPushButton("  Save Results")
        self.save_button.setIcon(self.create_icon("save"))
        self.save_button.setIconSize(QSize(20, 20))
        self.save_button.clicked.connect(self.save_results)
        file_layout.addWidget(self.save_button)
        
        file_group.setLayout(file_layout)
        left_panel.addWidget(file_group)
        
        # Processing operations
        processing_group = QGroupBox("Image Processing")
        processing_layout = QVBoxLayout()
        
        # Segment button with icon
        self.segment_button = QPushButton("  Perform Segmentation")
        self.segment_button.setIcon(self.create_icon("split"))
        self.segment_button.setIconSize(QSize(20, 20))
        self.segment_button.clicked.connect(self.segment_image)
        processing_layout.addWidget(self.segment_button)
        
        # Progress bar for segmentation
        self.process_progress = QProgressBar()
        self.process_progress.setRange(0, 100)
        self.process_progress.setValue(0)
        self.process_progress.setTextVisible(True)
        self.process_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 5px;
            }
        """)
        processing_layout.addWidget(self.process_progress)
        
        # Classify button with icon
        self.classify_button = QPushButton("  Classify White Cell")
        self.classify_button.setIcon(self.create_icon("tag"))
        self.classify_button.setIconSize(QSize(20, 20))
        self.classify_button.clicked.connect(self.classify_white_cell)
        processing_layout.addWidget(self.classify_button)
        
        processing_group.setLayout(processing_layout)
        left_panel.addWidget(processing_group)
        
        # Enhancement operations
        enhancement_group = QGroupBox("Image Enhancement")
        enhancement_layout = QVBoxLayout()
        
        # Contrast enhancement with styled slider
        contrast_layout = QHBoxLayout()
        contrast_icon = QLabel()
        contrast_icon.setPixmap(self.create_icon_pixmap("contrast", QSize(20, 20)))
        contrast_layout.addWidget(contrast_icon)
        
        contrast_label = QLabel("Contrast:")
        contrast_label.setStyleSheet("font-weight: bold;")
        contrast_layout.addWidget(contrast_label)
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(0)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(50)
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)
        contrast_layout.addWidget(self.contrast_slider)
        
        self.contrast_value = QLabel("1.0")
        contrast_layout.addWidget(self.contrast_value)
        enhancement_layout.addLayout(contrast_layout)
        
        # Sharpening with styled slider
        sharpening_layout = QHBoxLayout()
        sharpening_icon = QLabel()
        sharpening_icon.setPixmap(self.create_icon_pixmap("sharpen", QSize(20, 20)))
        sharpening_layout.addWidget(sharpening_icon)
        
        sharpening_label = QLabel("Sharpening:")
        sharpening_label.setStyleSheet("font-weight: bold;")
        sharpening_layout.addWidget(sharpening_label)
        
        self.sharpening_slider = QSlider(Qt.Horizontal)
        self.sharpening_slider.setMinimum(0)
        self.sharpening_slider.setMaximum(100)
        self.sharpening_slider.setValue(0)
        self.sharpening_slider.valueChanged.connect(self.apply_sharpening)
        sharpening_layout.addWidget(self.sharpening_slider)
        
        self.sharpening_value = QLabel("0.0")
        sharpening_layout.addWidget(self.sharpening_value)
        enhancement_layout.addLayout(sharpening_layout)
        
        # Histogram button with icon
        self.histogram_button = QPushButton("  Show Histogram")
        self.histogram_button.setIcon(self.create_icon("chart-bar"))
        self.histogram_button.setIconSize(QSize(20, 20))
        self.histogram_button.clicked.connect(self.show_histogram)
        enhancement_layout.addWidget(self.histogram_button)
        
        enhancement_group.setLayout(enhancement_layout)
        left_panel.addWidget(enhancement_group)
        
        # Results display with styled box
        results_group = QGroupBox("Classification Results")
        results_layout = QVBoxLayout()
        
        self.results_label = QLabel("No classification results yet.")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("""
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            font-weight: bold;
        """)
        self.results_label.setMinimumHeight(60)
        results_layout.addWidget(self.results_label)
        
        results_group.setLayout(results_layout)
        left_panel.addWidget(results_group)
        
        left_panel.addStretch()

        # Group Information
        group_info_group = QGroupBox("Group Information")
        group_info_layout = QVBoxLayout()

        group_members = [
            "Group 5",
            "22110007 Nguyễn Nhật An",
            "22110028 Nguyễn Mai Huy Hoàng", 
            "22110070 Đinh Tô Quốc Thắng",
            "22110076 Trần Trung Tín"
        ]

        for member in group_members:
            member_label = QLabel(member)
            # member_label.setStyleSheet("font-size: 10px; color: #2c3e50;") # Optional: style as needed
            group_info_layout.addWidget(member_label)

        group_info_group.setLayout(group_info_layout)
        left_panel.addWidget(group_info_group)
        
        # Right panel - create container for image display
        right_container = QWidget()
        right_panel = QVBoxLayout(right_container)
        right_panel.setContentsMargins(10, 10, 10, 10)
        right_panel.setSpacing(15)
        
        # Add title to right panel
        image_title = QLabel("Image Visualization")
        image_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        image_title.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(image_title)
        
        # Right panel layout for images (3x1 grid)
        images_splitter = QSplitter(Qt.Vertical)
        images_splitter.setChildrenCollapsible(False)
        
        # Original image
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout()
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 200)
        self.original_image_label.setStyleSheet("""
            background-color: #f8f9fa;
            border: 2px solid #d4e2f4;
            border-radius: 5px;
        """)
        original_layout.addWidget(self.original_image_label)
        original_group.setLayout(original_layout)
        images_splitter.addWidget(original_group)
        
        # Segmentation mask
        mask_group = QGroupBox("Segmentation Mask")
        mask_layout = QVBoxLayout()
        self.mask_image_label = QLabel()
        self.mask_image_label.setAlignment(Qt.AlignCenter)
        self.mask_image_label.setMinimumSize(400, 200)
        self.mask_image_label.setStyleSheet("""
            background-color: #f8f9fa;
            border: 2px solid #d4e2f4;
            border-radius: 5px;
        """)
        mask_layout.addWidget(self.mask_image_label)
        mask_group.setLayout(mask_layout)
        images_splitter.addWidget(mask_group)
        
        # Segmented image
        segmented_group = QGroupBox("Overlay Segmented Image")
        segmented_layout = QVBoxLayout()
        self.segmented_image_label = QLabel()
        self.segmented_image_label.setAlignment(Qt.AlignCenter)
        self.segmented_image_label.setMinimumSize(400, 200)
        self.segmented_image_label.setStyleSheet("""
            background-color: #f8f9fa;
            border: 2px solid #d4e2f4;
            border-radius: 5px;
        """)
        segmented_layout.addWidget(self.segmented_image_label)
        segmented_group.setLayout(segmented_layout)
        images_splitter.addWidget(segmented_group)
        
        right_panel.addWidget(images_splitter)
        
        # Add panels to main splitter
        main_splitter.addWidget(left_container)
        main_splitter.addWidget(right_container)
        main_splitter.setStretchFactor(0, 1)  # Left panel
        main_splitter.setStretchFactor(1, 3)  # Right panel
        
        # Set central widget
        self.setCentralWidget(main_splitter)
    
    def create_icon(self, name):
        """Create an icon based on name (placeholder for actual icons)"""
        # In real app, use: return QIcon(f"icons/{name}.png")
        # Here we create a colored square as placeholder
        color_map = {
            "folder-open": QColor(255, 193, 7),  # Amber
            "save": QColor(0, 150, 136),  # Teal
            "split": QColor(156, 39, 176),  # Purple
            "tag": QColor(33, 150, 243),  # Blue
            "chart-bar": QColor(233, 30, 99),  # Pink
            "contrast": QColor(63, 81, 181),  # Indigo
            "sharpen": QColor(76, 175, 80),  # Green
            "microscope": QColor(96, 125, 139)  # Blue Grey
        }
        
        pixmap = QPixmap(24, 24)
        pixmap.fill(color_map.get(name, QColor(120, 120, 120)))
        return QIcon(pixmap)
        
    def create_icon_pixmap(self, name, size):
        """Create a pixmap icon based on name (placeholder for actual icons)"""
        color_map = {
            "folder-open": QColor(255, 193, 7),
            "save": QColor(0, 150, 136),
            "split": QColor(156, 39, 176),
            "tag": QColor(33, 150, 243),
            "chart-bar": QColor(233, 30, 99),
            "contrast": QColor(63, 81, 181),
            "sharpen": QColor(76, 175, 80),
            "microscope": QColor(96, 125, 139)
        }
        
        pixmap = QPixmap(size)
        pixmap.fill(color_map.get(name, QColor(120, 120, 120)))
        return pixmap
        
    def load_image(self):
        """Load an image from the file system"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            # Show loading in status bar
            self.statusBar.showMessage("Loading image...")
            
            # Load image
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                self.results_label.setText("Failed to load image!")
                self.statusBar.showMessage("Error loading image", 3000)
                return
                
            # Convert BGR to RGB (OpenCV loads as BGR)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            self.display_image(self.original_image, self.original_image_label)
            
            # Reset other displays
            self.mask_image_label.clear()
            self.segmented_image_label.clear()
            self.results_label.setText("Image loaded successfully. Ready for processing.")
            
            # Update status bar
            self.statusBar.showMessage(f"Image loaded: {file_path.split('/')[-1]} ({self.original_image.shape[1]}x{self.original_image.shape[0]})", 5000)
            
            # Reset sliders to default
            self.contrast_slider.setValue(50)
            self.contrast_value.setText("1.0")
            self.sharpening_slider.setValue(0)
            self.sharpening_value.setText("0.0")
            self.process_progress.setValue(0)
    
    def segment_image(self):
        """Perform image segmentation using the pre-trained U-Net model."""
        if self.original_image is None:
            self.results_label.setText("Please load an image first!")
            self.statusBar.showMessage("Error: No image loaded", 3000)
            return

        # Show progress and status
        self.statusBar.showMessage("Segmenting image...")
        self.process_progress.setValue(10)
        QApplication.processEvents()

        # Load the segmentation model
        try:
            # Consider loading the model once during __init__ if it's always the same
            seg_model = load_model("models/unet_model.h5")
        except Exception as e:
            self.results_label.setText("Failed to load segmentation model!")
            self.statusBar.showMessage(f"Error: {e}", 3000)
            return

        # Preprocess the image
        img_resized = cv2.resize(self.original_image, (256, 256))
        input_tensor = img_resized / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Predict the mask
        try:
            mask_pred = seg_model.predict(input_tensor)[0, :, :, 0]
            mask_bin = (mask_pred > 0.5).astype(np.uint8)
        except Exception as e:
            self.results_label.setText("Segmentation failed!")
            self.statusBar.showMessage(f"Error: {e}", 3000)
            return

        # Resize mask back to original size
        self.segmented_mask = cv2.resize(mask_bin, (self.original_image.shape[1], self.original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create an overlay for the segmented region
        overlay = self.original_image.copy()
        # Define color for segmentation (e.g., red)
        segment_color = np.array([255, 0, 0], dtype=np.uint8) 
        
        # Apply color to the segmented region in the overlay
        # Ensure segmented_mask is boolean or 0/1 for indexing
        overlay[self.segmented_mask == 1] = segment_color
        
        # Blend the original image with the overlay
        alpha = 0.4  # Opacity factor (0.0 transparent, 1.0 opaque)
        self.segmented_image = cv2.addWeighted(overlay, alpha, self.original_image, 1 - alpha, 0)

        # Display results
        # Ensure mask is displayed as a grayscale image
        self.display_image(self.segmented_mask * 255, self.mask_image_label) 
        self.display_image(self.segmented_image, self.segmented_image_label)

        # Update status and results
        self.results_label.setText("Segmentation completed. Ready for classification.")
        self.statusBar.showMessage("Segmentation completed", 5000)
        self.process_progress.setValue(100)
    
    def classify_white_cell(self):
        """Classify the type of white blood cell using the pre-trained classifier."""
        if self.segmented_mask is None:
            self.results_label.setText("Please perform segmentation first!")
            self.statusBar.showMessage("Error: No segmentation performed", 3000)
            return

        # Extract features from the segmented region
        self.statusBar.showMessage("Extracting features...")
        self.process_progress.setValue(30)
        QApplication.processEvents()

        try:
            from utils.features import extract_features # Consider moving this import to the top of the file
            features = extract_features(self.original_image, self.segmented_mask)
        except ImportError:
            self.results_label.setText("Failed to import 'extract_features' from utils!")
            self.statusBar.showMessage("Error: Missing 'extract_features' utility.", 3000)
            return
        except Exception as e:
            self.results_label.setText(f"Failed to extract features! Error: {e}")
            self.statusBar.showMessage(f"Error: {e}", 3000)
            return

        # Load the classifier model
        try:
            # Construct path relative to the script file for robustness
            script_dir = os.path.dirname(os.path.abspath(__file__))
            classifier_model_path = os.path.join(script_dir, "models", "classifier.pkl")
            
            if not os.path.exists(classifier_model_path):
                self.results_label.setText(f"Classifier model file not found: {classifier_model_path}")
                self.statusBar.showMessage(f"Error: File not found at {classifier_model_path}", 5000)
                return

            clf_model = joblib.load(classifier_model_path)
        except Exception as e:
            # Updated error message to include the specific exception
            self.results_label.setText(f"Failed to load classifier model! Error: {e}") 
            self.statusBar.showMessage(f"Error loading classifier: {e}", 5000)
            return

        # Predict the cell type
        self.statusBar.showMessage("Classifying white blood cell...")
        self.process_progress.setValue(60)
        QApplication.processEvents()

        try:
            pred_label = clf_model.predict([features])[0]
        except Exception as e:
            self.results_label.setText("Classification failed!")
            self.statusBar.showMessage(f"Error: {e}", 3000)
            return

        # Display the result
        result_html = f"""
        <div style='text-align:center;'>
            <h3 style='color:#2980b9;'>Classification Result:</h3>
            <p style='font-size:16px; font-weight:bold; color:#16a085;'>{pred_label} White Blood Cell</p>
        </div>
        """
        self.results_label.setText(result_html)
        self.results_label.setTextFormat(Qt.RichText)
        self.statusBar.showMessage(f"Classification complete: {pred_label}", 5000)
        self.process_progress.setValue(100)
    
    def adjust_contrast(self):
        """Adjust image contrast"""
        if self.original_image is None:
            return
        
        value = self.contrast_slider.value() / 50.0  # Scale to range [0, 2]
        self.contrast_value.setText(f"{value:.1f}")
        
        # Apply contrast adjustment
        adjusted = exposure.adjust_gamma(self.original_image, value)
        
        # Update display
        self.display_image(adjusted, self.original_image_label)
        self.statusBar.showMessage(f"Contrast adjusted to {value:.1f}", 2000)
    
    def apply_sharpening(self):
        """Apply highpass filter to sharpen the image"""
        if self.original_image is None:
            return
        
        # Get slider value and convert to sharpening strength
        strength = self.sharpening_slider.value() / 100.0
        self.sharpening_value.setText(f"{strength:.1f}")
        
        if strength == 0:
            self.display_image(self.original_image, self.original_image_label)
            return
        
        # Convert to grayscale if needed
        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = self.original_image.copy()
        
        # Apply unsharp masking (a common sharpening technique)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sharpened = cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)
        
        # If original is color, convert back to RGB
        if len(self.original_image.shape) == 3:
            # Create colored sharpened image
            sharpened_rgb = self.original_image.copy()
            for i in range(3):
                channel = self.original_image[:,:,i]
                blurred_channel = cv2.GaussianBlur(channel, (5, 5), 0)
                sharpened_rgb[:,:,i] = cv2.addWeighted(channel, 1 + strength, blurred_channel, -strength, 0)
            
            self.display_image(sharpened_rgb, self.original_image_label)
        else:
            self.display_image(sharpened, self.original_image_label)
        
        self.statusBar.showMessage(f"Sharpening applied with strength {strength:.1f}", 2000)
    
    def show_histogram(self):
        """Display histogram of the original image"""
        if self.original_image is None:
            self.results_label.setText("Please load an image first!")
            self.statusBar.showMessage("Error: No image loaded", 3000)
            return
        
        # Create a new window for histogram with styled appearance
        histogram_window = QWidget()
        histogram_window.setWindowTitle("Image Histogram Analysis")
        histogram_window.setGeometry(300, 300, 800, 600)
        histogram_window.setStyleSheet("""
            QWidget {
                background-color: #f0f4f8;
                border: 2px solid #6c8ebf;
                border-radius: 10px;
            }
            QLabel {
                color: #2c3e50;
                font-weight: bold;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        # Create layout for the window
        layout = QVBoxLayout()
        
        # Add title
        title_label = QLabel("Image Histogram Analysis")
        title_label.setStyleSheet("font-size: 16px; color: #2c3e50; text-align: center;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create a matplotlib figure with custom style
        plt.style.use('seaborn-v0_8-whitegrid')
        figure = plt.figure(figsize=(8, 6))
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        
        # Add information panel
        info_layout = QHBoxLayout()
        
        # Create stats for the image
        stats_group = QGroupBox("Image Statistics")
        stats_layout = QVBoxLayout()
        
        if len(self.original_image.shape) == 3:  # Color image
            # Calculate stats for each channel
            for i, color_name in enumerate(['Red', 'Green', 'Blue']):
                channel = self.original_image[:,:,i]
                mean_val = np.mean(channel)
                std_val = np.std(channel)
                min_val = np.min(channel)
                max_val = np.max(channel)
                
                stats_text = QLabel(f"{color_name} Channel: Mean={mean_val:.1f}, Std={std_val:.1f}, Min={min_val}, Max={max_val}")
                stats_layout.addWidget(stats_text)
        else:  # Grayscale
            mean_val = np.mean(self.original_image)
            std_val = np.std(self.original_image)
            min_val = np.min(self.original_image)
            max_val = np.max(self.original_image)
            
            stats_text = QLabel(f"Grayscale: Mean={mean_val:.1f}, Std={std_val:.1f}, Min={min_val}, Max={max_val}")
            stats_layout.addWidget(stats_text)
            
        stats_group.setLayout(stats_layout)
        info_layout.addWidget(stats_group)
        
        layout.addLayout(info_layout)
        
        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(histogram_window.close)
        layout.addWidget(close_button)
        
        histogram_window.setLayout(layout)
        
        # Plot histogram with enhanced styling
        plt.clf()
        
        if len(self.original_image.shape) == 3:  # Color image
            colors = ('r', 'g', 'b')
            color_names = ('Red', 'Green', 'Blue')
            
            for i, (color, name) in enumerate(zip(colors, color_names)):
                hist = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
                plt.plot(hist, color=color, linewidth=2, alpha=0.7, label=name)
            
            plt.title('RGB Color Histogram', fontsize=14, fontweight='bold')
            plt.legend(loc='upper right')
            
        else:  # Grayscale image
            hist = cv2.calcHist([self.original_image], [0], None, [256], [0, 256])
            plt.plot(hist, color='gray', linewidth=2)
            plt.fill_between(range(256), hist.flatten(), alpha=0.3, color='gray')
            plt.title('Grayscale Histogram', fontsize=14, fontweight='bold')
        
        plt.xlabel('Pixel Intensity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim([0, 256])
        
        # Add vertical lines for mean value
        if len(self.original_image.shape) == 3:
            for i, color in enumerate(colors):
                mean_val = np.mean(self.original_image[:,:,i])
                plt.axvline(x=mean_val, color=color, linestyle='--', 
                           label=f'{color_names[i]} Mean: {mean_val:.1f}')
        else:
            mean_val = np.mean(self.original_image)
            plt.axvline(x=mean_val, color='black', linestyle='--', 
                       label=f'Mean: {mean_val:.1f}')
        
        plt.tight_layout()
        
        # Show the window
        canvas.draw()
        histogram_window.show()
        
        self.statusBar.showMessage("Histogram analysis displayed", 3000)
    
    def save_results(self):
        """Save the segmented image and classification results"""
        if self.segmented_image is None:
            self.results_label.setText("No results to save. Perform segmentation first!")
            self.statusBar.showMessage("Error: No results to save", 3000)
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "Image Files (*.png)")
        
        if file_path:
            # Show saving status
            self.statusBar.showMessage("Saving results...")
            self.process_progress.setValue(30)
            QApplication.processEvents()
            
            # Save segmented image
            cv2.imwrite(file_path, cv2.cvtColor(self.segmented_image, cv2.COLOR_RGB2BGR))
            self.process_progress.setValue(60)
            QApplication.processEvents()
            
            # Save text results
            text_file = os.path.splitext(file_path)[0] + "_results.txt"
            with open(text_file, 'w') as f:
                # Strip HTML if present
                plain_text = self.results_label.text()
                if "<" in plain_text and ">" in plain_text:
                    plain_text = "Classification Result: " + plain_text.split("<p style=")[1].split(">")[1].split("<")[0]
                f.write(plain_text)
            
            # Save mask image
            mask_file = os.path.splitext(file_path)[0] + "_mask.png"
            cv2.imwrite(mask_file, cv2.cvtColor(self.segmented_mask, cv2.COLOR_RGB2BGR))
            
            self.process_progress.setValue(100)
            QApplication.processEvents()
            
            success_msg = f"Results saved to {file_path}"
            self.results_label.setText(success_msg)
            self.statusBar.showMessage(success_msg, 5000)
            
            # Reset progress bar after a delay
            QTimer.singleShot(2000, lambda: self.process_progress.setValue(0))
    
    def display_image(self, image, label):
        """Display an image on a QLabel with enhanced visual style"""
        if image is None:
            return
        
        # Convert numpy array to QImage
        if len(image.shape) == 3:  # Color image
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:  # Grayscale image
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Convert QImage to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale pixmap to fit the label while maintaining aspect ratio
        label_size = label.size()
        pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set pixmap to label
        label.setPixmap(pixmap)
        
        # Add border and shadow effect
        label.setStyleSheet("""
            background-color: #f8f9fa; 
            border: 2px solid #d4e2f4;
            border-radius: 5px;
            padding: 5px;
        """)

def main():
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = MedicalImageSegmentationApp()
    window.show()
    
    # Display welcome message
    window.statusBar.showMessage("Welcome to Medical Image Segmentation and Classification System", 5000)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()