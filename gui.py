import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QFileDialog, QComboBox, QSlider, QGroupBox,
                            QStatusBar, QSplitter, QProgressBar, QFrame, QToolButton, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor, QLinearGradient, QBrush, QPainter 
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
import joblib
import torch
from torchvision import transforms
from utils.features import extract_features
from unet import UNet

class MedicalImageSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set application style and colors
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f7f9fc;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #c0d3e8;
                border-radius: 8px;
                margin-top: 1ex;
                background-color: rgba(233, 241, 251, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 8px;
                color: #2c3e50;
                background-color: #e1ebf7;
                border-radius: 1px;
            }
            QPushButton {
                background-color: #2081e2;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
                min-height: 30px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1669c4;
            }
            QPushButton:pressed {
                background-color: #0d4e96;
            }
            QPushButton:disabled {
                background-color: #a0b5cc;
                color: #e5e5e5;
            }
            QLabel {
                color: #2c3e50;
            }
            QSlider::groove:horizontal {
                border: 1px solid #c0d3e8;
                background: white;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2081e2, stop:1 #1669c4);
                border: none;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 1px solid #c0d3e8;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f4f8;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2081e2, stop:1 #49aeff);
                border-radius: 5px;
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
        
        # Load PyTorch model
        try:
            self.segmentation_model = UNet(n_channels=3, n_classes=1)
            self.segmentation_model.load_state_dict(
                torch.load("models/unet_best-8-5-16-26.pth", map_location=torch.device('cpu'))
            )
            self.segmentation_model.eval()
            print("PyTorch model loaded successfully!")
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            self.segmentation_model = None
        
    def load_classifier(self):
        """
        In real application, load a pre-trained classifier for white cell classification
        Here we mock up a simple classifier
        """
        class MockClassifier:
            def predict(self, features):
                # For demo purposes, return a random class
                cell_types = ['Neutrophil', 'Lymphocyte', 'Monocyte', 'Eosinophil', 'Basophil']
                return [np.random.choice(cell_types)]
        
        return MockClassifier()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle("Medical Image Segmentation and Classification")
        self.setGeometry(100, 100, 1500, 950)
        
        # Set application font
        font = QFont("Segoe UI", 10)
        QApplication.setFont(font)
        
        # Main container widget
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top header with fixed height - important fix
        top_header = QFrame()
        top_header.setStyleSheet("""
            QFrame {
                background-color: #e6eef5;
                border-bottom: 2px solid #c0d3e8;
            }
        """)
        # Set fixed height instead of percentage
        top_header.setFixedHeight(300)
        
        # Top header layout
        top_header_layout = QVBoxLayout(top_header)
        top_header_layout.setContentsMargins(15, 10, 15, 10)
        
        # Create the app title banner with fixed height
        header_frame = QFrame()
        header_frame.setFixedHeight(100)  # Fixed height for the banner
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                            stop:0 #1a5276, stop:1 #2081e2); 
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 5, 15, 5)
        header_layout.setStretch(0, 2)  
        header_layout.setStretch(1, 1)  
        header_layout.setStretch(2, 3)
        
        # App title and subtitle
        title_layout = QVBoxLayout()
        app_title = QLabel("Blood Cell Analyzer")
        app_title.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            color: white;
            background: transparent;
            text-align: center;
        """)
        app_subtitle = QLabel("Medical Image Segmentation & Classification")
        app_subtitle.setStyleSheet("""
            font-size: 12px; 
            color: #e6eef5;
            background: transparent;
            text-align: center;
        """)
        title_layout.addWidget(app_title)
        title_layout.addWidget(app_subtitle)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # Add team information to the right of header
        team_layout = QVBoxLayout()
        team_layout.setContentsMargins(0, 0, 10, 0)
        team_name = QLabel("Group 4")
        team_name.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            color: white; 
            background: transparent;
        """)
        team_name.setAlignment(Qt.AlignRight)
        team_layout.addWidget(team_name)
        
        # Add team members
        # Create a grid layout for team members (2 columns)
        members_grid = QGridLayout()
        members_grid.setSpacing(5)  # Spacing between cells
        members_grid.setContentsMargins(0, 0, 0, 0)  # Minimize margins

        # Member information
        members = [
            "22110007 - Nguyễn Nhật An",
            "22110028 - Nguyễn Mai Huy Hoàng",
            "22110070 - Đinh Tô Quốc Thắng",
            "22110076 - Trần Trung Tín"
        ]

        # Common style for all member labels
        member_style = """
            font-size: 12px;
            font-weight: 500; 
            color: #e6eef5; 
            background: transparent;
            padding: 0px;
            margin: 0px;
            border: none;
        """

        # Add members to the grid - 2 columns
        for i, member in enumerate(members):
            row = i // 2  # Integer division to determine row
            col = i % 2   # Modulo to determine column (0 or 1)
            
            member_label = QLabel(member)
            member_label.setStyleSheet(member_style)
            member_label.setAlignment(Qt.AlignLeft)
            
            members_grid.addWidget(member_label, row, col)

        # Add the grid to the team layout
        team_layout.addLayout(members_grid)
        header_layout.addLayout(team_layout)
        top_header_layout.addWidget(header_frame)
        
        # Create horizontal layout for controls (previously in left sidebar)
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        # File operations group with fixed height
        file_group = QGroupBox("File Operations")
        file_group.setFixedHeight(150)  # Fixed height
        file_layout = QHBoxLayout()
        file_layout.setContentsMargins(10, 15, 10, 10)
        file_layout.setSpacing(10)
        
        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.setMinimumHeight(40)
        self.load_button.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_button)
        
        # Save results button
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        file_layout.addWidget(self.save_button)
        
        file_group.setLayout(file_layout)
        controls_layout.addWidget(file_group)
        
        # Processing operations group with fixed height
        processing_group = QGroupBox("Image Processing")
        processing_group.setFixedHeight(150)  # Fixed height
        processing_layout = QHBoxLayout()
        processing_layout.setContentsMargins(10, 15, 10, 10)
        processing_layout.setSpacing(10)
        
        # Segment button
        self.segment_button = QPushButton("Perform Segmentation")
        self.segment_button.clicked.connect(self.segment_image)
        processing_layout.addWidget(self.segment_button)
        
        # Classify button
        self.classify_button = QPushButton("Classify White Cell")
        self.classify_button.clicked.connect(self.classify_white_cell)
        processing_layout.addWidget(self.classify_button)
        
        processing_group.setLayout(processing_layout)
        controls_layout.addWidget(processing_group)
        
        # Results display group with fixed height
        results_group = QGroupBox("Classification Results")
        results_group.setFixedHeight(150)  # Fixed height is important
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(10, 15, 10, 10)
        
        # Results label in a card with fixed height content
        result_card = QFrame()
        result_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #d4e2f4;
            }
        """)
        result_card_layout = QVBoxLayout(result_card)
        result_card_layout.setContentsMargins(15, 5, 15, 5)
        
        self.results_label = QLabel("No classification results yet.")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("""
            padding: 5px;
            font-weight: bold;
            color: #2c3e50;
            background: transparent;
        """)
        self.results_label.setFixedHeight(110)
        self.results_label.setAlignment(Qt.AlignCenter)
        result_card_layout.addWidget(self.results_label)
        
        results_layout.addWidget(result_card)
        results_group.setLayout(results_layout)
        controls_layout.addWidget(results_group)
        
        top_header_layout.addLayout(controls_layout)
        
        # Add top header to main layout
        main_layout.addWidget(top_header)
        
        # Main window content area
        main_content = QWidget()
        main_content.setStyleSheet("""
            background-color: #f7f9fc;
        """)
        
        # Create the main content layout for images (horizontal)
        main_content_layout = QHBoxLayout(main_content)
        main_content_layout.setContentsMargins(15, 15, 15, 15)
        main_content_layout.setSpacing(15)
        
        # Original image
        original_card = QGroupBox("Original Image")
        original_card.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #c0d3e8;
                border-radius: 8px;
                margin-top: 1ex;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 8px;
                color: #2c3e50;
                background-color: white;
                border-radius: 4px;
            }
        """)
        
        original_layout = QVBoxLayout()
        original_layout.setContentsMargins(10, 15, 10, 10)
        
        # Image frame with shadow
        image_frame = QFrame()
        image_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e1ebf7;
                border-radius: 5px;
            }
        """)
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(5, 5, 5, 5)
        
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFixedSize(500, 500)
        self.original_image_label.setStyleSheet("background: transparent;")
        image_layout.addWidget(self.original_image_label)
        
        original_layout.addWidget(image_frame)
        original_card.setLayout(original_layout)
        main_content_layout.addWidget(original_card)
        
        # Segmentation mask
        mask_group = QGroupBox("Segmentation Mask")
        mask_layout = QVBoxLayout()
        mask_layout.setContentsMargins(10, 15, 10, 10)
        
        mask_frame = QFrame()
        mask_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e1ebf7;
                border-radius: 5px;
            }
        """)
        mask_frame_layout = QVBoxLayout(mask_frame)
        mask_frame_layout.setContentsMargins(5, 5, 5, 5)
        
        self.mask_image_label = QLabel()
        self.mask_image_label.setAlignment(Qt.AlignCenter)
        self.mask_image_label.setFixedSize(500, 500)
        self.mask_image_label.setStyleSheet("background: transparent;")
        mask_frame_layout.addWidget(self.mask_image_label)
        
        mask_layout.addWidget(mask_frame)
        mask_group.setLayout(mask_layout)
        main_content_layout.addWidget(mask_group)
        
        # Segmented image
        segmented_group = QGroupBox("Overlay Segmented Image")
        segmented_layout = QVBoxLayout()
        segmented_layout.setContentsMargins(10, 15, 10, 10)
        
        segmented_frame = QFrame()
        segmented_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e1ebf7;
                border-radius: 5px;
            }
        """)
        segmented_frame_layout = QVBoxLayout(segmented_frame)
        segmented_frame_layout.setContentsMargins(5, 5, 5, 5)
        
        self.segmented_image_label = QLabel()
        self.segmented_image_label.setAlignment(Qt.AlignCenter)
        self.segmented_image_label.setFixedSize(500, 500)
        self.segmented_image_label.setStyleSheet("background: transparent;")
        segmented_frame_layout.addWidget(self.segmented_image_label)
        
        segmented_layout.addWidget(segmented_frame)
        segmented_group.setLayout(segmented_layout)
        main_content_layout.addWidget(segmented_group)
        
        # Add main content to main layout
        main_layout.addWidget(main_content)
        
        # Set the proportion between top header and main content
        main_layout.setStretch(0, 0)  # Top header (fixed height, no stretch)
        main_layout.setStretch(1, 1)  # Main content (takes remaining space)
        
        # Set central widget
        self.setCentralWidget(main_container)
    
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
        """Create a professional-looking icon"""
        
        # Tạo gradient pixmap cho logo app
        if name == "microscope":
            pixmap = QPixmap(size)
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Tạo gradient
            gradient = QLinearGradient(0, 0, size.width(), size.height())
            gradient.setColorAt(0, QColor(32, 129, 226))    # Blue
            gradient.setColorAt(1, QColor(20, 170, 192))    # Teal
            
            # Vẽ biểu tượng kính hiển vi đơn giản
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            
            # Vẽ phần thân kính hiển vi - chuyển đổi tất cả giá trị float thành int
            painter.drawRoundedRect(int(size.width()*0.25), int(size.height()*0.4), 
                                int(size.width()*0.5), int(size.height()*0.5), 5, 5)
            
            # Vẽ phần ống kính - chuyển đổi tất cả giá trị float thành int
            painter.drawEllipse(int(size.width()*0.3), int(size.height()*0.15), 
                            int(size.width()*0.4), int(size.width()*0.4))
            
            # Vẽ phần chân đế - chuyển đổi tất cả giá trị float thành int
            painter.drawRoundedRect(int(size.width()*0.15), int(size.height()*0.85), 
                                int(size.width()*0.7), int(size.height()*0.1), 3, 3)
            
            painter.end()
            return pixmap
        
        # Tạo gradient icons cho các nút
        color_map = {
            "folder-open": [QColor(52, 152, 219), QColor(41, 128, 185)],
            "save": [QColor(46, 204, 113), QColor(39, 174, 96)],
            "split": [QColor(155, 89, 182), QColor(142, 68, 173)],
            "tag": [QColor(52, 152, 219), QColor(41, 128, 185)],
            "chart-bar": [QColor(231, 76, 60), QColor(192, 57, 43)],
            "contrast": [QColor(52, 73, 94), QColor(44, 62, 80)],
            "sharpen": [QColor(46, 204, 113), QColor(39, 174, 96)]
        }
        
        pixmap = QPixmap(size)
        pixmap.fill(Qt.transparent)
        
        if name in color_map:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Tạo gradient
            gradient = QLinearGradient(0, 0, 0, size.height())
            gradient.setColorAt(0, color_map[name][0])
            gradient.setColorAt(1, color_map[name][1])
            
            # Vẽ biểu tượng với gradient
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(0, 0, size.width(), size.height(), 8, 8)
            painter.end()
        else:
            pixmap.fill(QColor(120, 120, 120))
            
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
    
    def segment_image(self):
        """Perform image segmentation using the PyTorch model."""
        if self.original_image is None:
            self.results_label.setText("Please load an image first!")
            self.statusBar.showMessage("Error: No image loaded", 3000)
            return

        if self.segmentation_model is None:
            self.results_label.setText("Segmentation model not loaded!")
            self.statusBar.showMessage("Error: Model not loaded", 3000)
            return

        # Show progress and status
        self.statusBar.showMessage("Segmenting image...")
        QApplication.processEvents()

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust based on model training
        ])
        input_tensor = preprocess(self.original_image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        try:
            with torch.no_grad():
                output = self.segmentation_model(input_tensor)
                mask_pred = torch.sigmoid(output).squeeze().numpy()  # Convert to numpy array
                mask_bin = (mask_pred > 0.5).astype(np.uint8)  # Binarize the mask
        except Exception as e:
            self.results_label.setText("Segmentation failed!")
            self.statusBar.showMessage(f"Error: {e}", 3000)
            return

        # Resize mask back to original size
        self.segmented_mask = cv2.resize(mask_bin, (self.original_image.shape[1], self.original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create an overlay for the segmented region
        overlay = self.original_image.copy()
        segment_color = np.array([255, 0, 0], dtype=np.uint8)  # Red color for segmentation
        overlay[self.segmented_mask == 1] = segment_color

        # Blend the original image with the overlay
        alpha = 0.4
        self.segmented_image = cv2.addWeighted(overlay, alpha, self.original_image, 1 - alpha, 0)

        # Display results
        self.display_image(self.segmented_mask * 255, self.mask_image_label)
        self.display_image(self.segmented_image, self.segmented_image_label)

        # Update status and results
        self.results_label.setText("Segmentation completed. Ready for classification.")
        self.statusBar.showMessage("Segmentation completed", 5000)
    
    def classify_white_cell(self):
        """Classify the type of white blood cell using the pre-trained classifier."""
        if self.segmented_mask is None:
            self.results_label.setText("Please perform segmentation first!")
            self.statusBar.showMessage("Error: No segmentation performed", 3000)
            return

        # Extract features from the segmented region
        self.statusBar.showMessage("Extracting features...")
        QApplication.processEvents()

        try:
            # Consider moving this import to the top of the file
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
        QApplication.processEvents()

        try:
            pred_label = clf_model.predict([features])[0]
            pred_probs = clf_model.predict_proba([features])[0]
            accuracy = np.max(pred_probs) * 100 
        except Exception as e:
            self.results_label.setText("Classification failed!")
            self.statusBar.showMessage(f"Error: {e}", 3000)
            return

        # Display the result - shortened to fit fixed height
        result_html = f"""
        <div style='text-align:center;'>
            <p style='font-size:14px; font-weight:bold; color:#2980b9;'>Classification Result:</p>
            <p style='font-size:14px; font-weight:bold; color:#16a085;'>{pred_label} White Blood Cell</p>
            <p style='font-size:12px; color:#2c3e50;'>Accuracy: {accuracy:.2f}%</p>
        </div>
        """
        self.results_label.setText(result_html)
        self.results_label.setTextFormat(Qt.RichText)
        self.statusBar.showMessage(f"Classification complete: {pred_label}", 5000)
    
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
            QApplication.processEvents()
            
            # Save segmented image
            cv2.imwrite(file_path, cv2.cvtColor(self.segmented_image, cv2.COLOR_RGB2BGR))
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
            cv2.imwrite(mask_file, self.segmented_mask * 255)
            
            QApplication.processEvents()
            
            success_msg = f"Results saved to {file_path}"
            self.results_label.setText(success_msg)
            self.statusBar.showMessage(success_msg, 5000)
    
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
