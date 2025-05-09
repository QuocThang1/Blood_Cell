import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QFileDialog, QComboBox, QSlider, QGroupBox,
                            QStatusBar, QSplitter, QProgressBar, QFrame, QToolButton, QTabWidget, QTextEdit, QMessageBox)


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
import joblib # Add this import
import torch
from torchvision import transforms
from utils.features import extract_features

from unet import UNet  # Ensure the file is named unet.py

class MedicalImageSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # # Set window properties
        
        

        
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
        
        # Load PyTorch model
        try:
            self.segmentation_model = UNet(n_channels=3, n_classes=1)  # Specify the number of input channels and output classes
            self.segmentation_model.load_state_dict(
                torch.load("models/unet_best-8-5-16-26.pth", map_location=torch.device('cpu'))
            )
            self.segmentation_model.eval()  # Switch to evaluation mode
            print("PyTorch model loaded successfully!")
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            self.segmentation_model = None
        
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
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create widgets for tabs
        self.main_tab = QWidget()
        self.evaluation_tab = QWidget()
        
        # Setup main tab
        self.setup_main_tab()
        
        # Setup evaluation tab
        self.setup_evaluation_tab()
        
        # Add tabs to widget
        self.tab_widget.addTab(self.main_tab, "Single Image Analysis")
        self.tab_widget.addTab(self.evaluation_tab, "Model Evaluation")
        
        # Set as central widget (chỉ đặt một lần duy nhất)
        self.setCentralWidget(self.tab_widget)
        
        # Set application font
        font = QFont("Segoe UI", 10)
        QApplication.setFont(font)

    
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
        self.process_progress.setValue(10)
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
        self.process_progress.setValue(60)
        QApplication.processEvents()

        try:
            pred_label = clf_model.predict([features])[0]
            pred_probs = clf_model.predict_proba([features])[0]
            accuracy = np.max(pred_probs) * 100 
        except Exception as e:
            self.results_label.setText("Classification failed!")
            self.statusBar.showMessage(f"Error: {e}", 3000)
            return

        # Display the result
        result_html = f"""
        <div style='text-align:center;'>
            <h3 style='color:#2980b9;'>Classification Result:</h3>
            <p style='font-size:16px; font-weight:bold; color:#16a085;'>{pred_label} White Blood Cell</p>
            <p style='font-size:14px; color:#2c3e50;'>Accuracy: {accuracy:.2f}%</p>
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

    def setup_main_tab(self):
        # Create layout for main tab
        main_layout = QVBoxLayout(self.main_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
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
        
        # Add main splitter to the tab layout
        main_layout.addWidget(main_splitter)

    
    def setup_evaluation_tab(self):
        # Create layout for evaluation tab
        eval_layout = QVBoxLayout(self.evaluation_tab)
        eval_layout.setContentsMargins(10, 10, 10, 10)
        eval_layout.setSpacing(15)
        
        # Title for evaluation tab
        eval_title = QLabel("Model Evaluation and Analysis")
        eval_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        eval_title.setAlignment(Qt.AlignCenter)
        eval_layout.addWidget(eval_title)
        
        # Create top section for controls
        controls_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - dataset selection
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QVBoxLayout()
        
        # Dataset path display
        self.dataset_path_label = QLabel("No dataset selected")
        self.dataset_path_label.setWordWrap(True)
        self.dataset_path_label.setStyleSheet("""
            background-color: white;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #ccc;
        """)
        dataset_layout.addWidget(self.dataset_path_label)
        
        # Select dataset button
        select_dataset_btn = QPushButton("  Select Dataset Folder")
        select_dataset_btn.setIcon(self.create_icon("folder-open"))
        select_dataset_btn.setIconSize(QSize(20, 20))
        select_dataset_btn.clicked.connect(self.select_dataset_folder)
        dataset_layout.addWidget(select_dataset_btn)
        
        dataset_group.setLayout(dataset_layout)
        
        # Right side - evaluation controls
        eval_controls_group = QGroupBox("Evaluation Controls")
        eval_controls_layout = QVBoxLayout()
        
        # Segmentation evaluation button
        self.run_seg_eval_btn = QPushButton("  Evaluate Segmentation")
        self.run_seg_eval_btn.setIcon(self.create_icon("chart-bar"))
        self.run_seg_eval_btn.setIconSize(QSize(20, 20))
        self.run_seg_eval_btn.clicked.connect(self.evaluate_segmentation)
        self.run_seg_eval_btn.setEnabled(False)
        eval_controls_layout.addWidget(self.run_seg_eval_btn)
        
        # Classification evaluation button
        self.run_class_eval_btn = QPushButton("  Evaluate Classification")
        self.run_class_eval_btn.setIcon(self.create_icon("tag"))
        self.run_class_eval_btn.setIconSize(QSize(20, 20))
        self.run_class_eval_btn.clicked.connect(self.evaluate_classification)
        self.run_class_eval_btn.setEnabled(False)
        eval_controls_layout.addWidget(self.run_class_eval_btn)
        
        # Progress bar
        self.eval_progress = QProgressBar()
        self.eval_progress.setRange(0, 100)
        self.eval_progress.setValue(0)
        self.eval_progress.setTextVisible(True)
        self.eval_progress.setStyleSheet("""
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
        eval_controls_layout.addWidget(self.eval_progress)
        
        eval_controls_group.setLayout(eval_controls_layout)
        
        # Add groups to splitter
        controls_splitter.addWidget(dataset_group)
        controls_splitter.addWidget(eval_controls_group)
        
        eval_layout.addWidget(controls_splitter)
        
        # Results section
        results_splitter = QSplitter(Qt.Horizontal)
        
        # Text results
        text_results_group = QGroupBox("Evaluation Metrics")
        text_results_layout = QVBoxLayout()
        
        self.eval_results_text = QTextEdit()
        self.eval_results_text.setReadOnly(True)
        text_results_layout.addWidget(self.eval_results_text)
        
        text_results_group.setLayout(text_results_layout)
        
        # Chart results
        chart_results_group = QGroupBox("Visual Analysis")
        self.eval_chart_layout = QVBoxLayout()
        
        # Placeholder text
        placeholder_label = QLabel("Charts will appear here after evaluation")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: #888; font-style: italic;")
        self.eval_chart_layout.addWidget(placeholder_label)
        
        chart_results_group.setLayout(self.eval_chart_layout)
        
        # Add to splitter
        results_splitter.addWidget(text_results_group)
        results_splitter.addWidget(chart_results_group)
        
        eval_layout.addWidget(results_splitter, 1)  # Give it more stretch

    def select_dataset_folder(self):
        """Select a dataset folder containing image and mask subfolders"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        
        if folder_path:
            # Check if folder contains 'image' and 'mask' subfolders
            image_folder = os.path.join(folder_path, "image")
            mask_folder = os.path.join(folder_path, "mask")
            
            if not (os.path.exists(image_folder) and os.path.exists(mask_folder)):
                QMessageBox.warning(self, "Invalid Dataset", 
                                "The selected folder must contain 'image' and 'mask' subfolders.")
                return
            
            # Store folder paths
            self.dataset_folder = folder_path
            self.image_folder = image_folder
            self.mask_folder = mask_folder
            
            # Sort function for natural sorting
            def natural_sort_key(s):
                import re
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            
            # Get and sort image files
            self.image_files = [f for f in os.listdir(image_folder) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            self.image_files.sort(key=natural_sort_key)  # Sort naturally
            
            # Get and sort mask files
            self.mask_files = [f for f in os.listdir(mask_folder) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            self.mask_files.sort(key=natural_sort_key)  # Sort naturally
            
            # Check if we have matching counts
            valid_pairs = min(len(self.image_files), len(self.mask_files))
            
            if valid_pairs == 0:
                QMessageBox.warning(self, "No Valid Pairs", 
                                "No matching image-mask pairs found in the selected folders.")
                return
                
            # Update UI
            self.dataset_path_label.setText(f"Dataset: {folder_path}\n"
                                        f"Found {len(self.image_files)} images with {valid_pairs} valid image-mask pairs")
            
            # Enable evaluation button if valid pairs found
            self.run_seg_eval_btn.setEnabled(valid_pairs > 0)
            
            # Clear previous results
            self.eval_results_text.clear()
            self.clear_chart_layout()
            
            # Show information about the dataset
            self.statusBar.showMessage(f"Loaded dataset with {valid_pairs} image-mask pairs", 5000)


    def evaluate_segmentation(self):
        """Evaluate the segmentation model on the dataset"""
        if not hasattr(self, 'image_files') or not self.image_files:
            QMessageBox.warning(self, "No Images", "Please select a valid dataset folder first.")
            return
        
        if self.segmentation_model is None:
            QMessageBox.warning(self, "No Model", "Segmentation model is not loaded.")
            return
        
        # Clear previous results
        self.eval_results_text.clear()
        self.clear_chart_layout()
        
        # Initialize metrics
        results = {
            'image_name': [],
            'accuracy': [],
            'iou': [],
            'dice': [],
            'precision': [],
            'recall': []
        }
        
        # Update UI
        self.eval_results_text.append("Starting segmentation evaluation...")
        QApplication.processEvents()
        
        # Limit to 100 images
        max_images = min(100, min(len(self.image_files), len(self.mask_files)))
        
        # Process images
        for i in range(max_images):
            # Update progress
            self.eval_progress.setValue((i+1) * 100 // max_images)
            QApplication.processEvents()
            
            # Load image and ground truth mask using the sorted indices
            image_path = os.path.join(self.image_folder, self.image_files[i])
            mask_path = os.path.join(self.mask_folder, self.mask_files[i])
            
            # Load image and mask
            original_image = cv2.imread(image_path)
            ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if original_image is None or ground_truth is None:
                self.eval_results_text.append(f"Error loading image {self.image_files[i]} or mask {self.mask_files[i]}")
                continue
            
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            ground_truth_binary = (ground_truth > 0).astype(np.uint8)
            
            # Perform segmentation
            try:
                # Preprocess image
                preprocess = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                input_tensor = preprocess(original_image).unsqueeze(0)
                
                # Run model
                with torch.no_grad():
                    output = self.segmentation_model(input_tensor)
                    mask_pred = torch.sigmoid(output).squeeze().numpy()
                    predicted_mask = (mask_pred > 0.5).astype(np.uint8)
                
                # Resize mask to original size
                predicted_mask = cv2.resize(predicted_mask, 
                                        (original_image.shape[1], original_image.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                
                # Calculate metrics
                # Accuracy
                correct_pixels = np.sum(predicted_mask == ground_truth_binary)
                total_pixels = ground_truth_binary.size
                accuracy = correct_pixels / total_pixels
                
                # IoU and Dice
                intersection = np.sum(predicted_mask * ground_truth_binary)
                union = np.sum(predicted_mask) + np.sum(ground_truth_binary) - intersection
                
                iou = intersection / union if union > 0 else 0.0
                dice = (2. * intersection) / (np.sum(predicted_mask) + np.sum(ground_truth_binary)) if (np.sum(predicted_mask) + np.sum(ground_truth_binary)) > 0 else 0.0
                
                # Precision and Recall
                tp = intersection
                fp = np.sum(predicted_mask) - tp
                fn = np.sum(ground_truth_binary) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                # Store results
                results['image_name'].append(self.image_files[i])
                results['accuracy'].append(accuracy)
                results['iou'].append(iou)
                results['dice'].append(dice)
                results['precision'].append(precision)
                results['recall'].append(recall)
                    
            except Exception as e:
                self.eval_results_text.append(f"Error processing {self.image_files[i]}: {str(e)}")
        
        # Display results
        self.display_segmentation_results(results)
        
        # Enable classification evaluation
        self.run_class_eval_btn.setEnabled(len(results['dice']) > 0)

    def display_segmentation_results(self, results):
        """Display segmentation evaluation results with comprehensive charts"""
        if not results['dice']:
            self.eval_results_text.append("No valid results to display.")
            return
        
        # Calculate statistics
        mean_accuracy = np.mean(results['accuracy'])
        mean_iou = np.mean(results['iou'])
        mean_dice = np.mean(results['dice'])
        mean_precision = np.mean(results['precision'])
        mean_recall = np.mean(results['recall'])
        
        std_accuracy = np.std(results['accuracy'])
        std_iou = np.std(results['iou'])
        std_dice = np.std(results['dice'])
        std_precision = np.std(results['precision'])
        std_recall = np.std(results['recall'])
        
        # Display text results
        self.eval_results_text.append(f"\nSegmentation Evaluation Results:")
        self.eval_results_text.append(f"Number of images evaluated: {len(results['dice'])}")
        self.eval_results_text.append(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        self.eval_results_text.append(f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
        self.eval_results_text.append(f"Mean Dice Coefficient: {mean_dice:.4f} ± {std_dice:.4f}")
        self.eval_results_text.append(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
        self.eval_results_text.append(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
        
        # Create visual analysis with multiple plots
        
        # 1. Box plots for each metric
        fig1 = plt.figure(figsize=(8, 6))
        metrics = ['accuracy', 'iou', 'dice', 'precision', 'recall']
        metric_values = [results[m] for m in metrics]
        plt.boxplot(metric_values, labels=[m.capitalize() for m in metrics])
        plt.title('Distribution of Metrics')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        canvas1 = FigureCanvas(fig1)
        self.eval_chart_layout.addWidget(canvas1)
        
        # 2. Histogram of IoU scores with normal distribution overlay
        fig2 = plt.figure(figsize=(8, 6))
        plt.hist(results['iou'], bins=20, alpha=0.7, color='#3498db', edgecolor='black')
        plt.axvline(mean_iou, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean IoU: {mean_iou:.4f}')
        plt.title('IoU Score Distribution')
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        canvas2 = FigureCanvas(fig2)
        self.eval_chart_layout.addWidget(canvas2)
        
        # 3. Precision-Recall scatter plot 
        fig3 = plt.figure(figsize=(8, 6))
        plt.scatter(results['recall'], results['precision'], c=results['iou'], 
                cmap='viridis', alpha=0.7, edgecolors='w', linewidths=0.5)
        plt.colorbar(label='IoU Score')
        plt.title('Precision vs. Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        canvas3 = FigureCanvas(fig3)
        self.eval_chart_layout.addWidget(canvas3)
        
        # 4. Show best and worst samples
        if len(results['dice']) >= 6:  # Make sure we have enough samples
            # Find indices of best and worst samples
            best_indices = np.argsort(results['iou'])[-3:]
            worst_indices = np.argsort(results['iou'])[:3]
            
            # Create a figure to display them
            fig4 = plt.figure(figsize=(12, 8))
            plt.suptitle('Best and Worst Segmentation Results', fontsize=16)
            
            # Plot best samples
            for i, idx in enumerate(reversed(best_indices)):
                plt.subplot(2, 3, i + 1)
                plt.title(f"Best #{i+1}: IoU={results['iou'][idx]:.4f}")
                plt.text(0.5, 0.5, f"Image: {results['image_name'][idx]}", 
                        ha='center', va='center', wrap=True)
                plt.axis('off')
            
            # Plot worst samples
            for i, idx in enumerate(worst_indices):
                plt.subplot(2, 3, i + 4)
                plt.title(f"Worst #{i+1}: IoU={results['iou'][idx]:.4f}")
                plt.text(0.5, 0.5, f"Image: {results['image_name'][idx]}", 
                        ha='center', va='center', wrap=True)
                plt.axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            canvas4 = FigureCanvas(fig4)
            self.eval_chart_layout.addWidget(canvas4)


    def calculate_dice(self, y_true, y_pred):
        """Calculate Dice coefficient"""
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

    def calculate_iou(self, y_true, y_pred):
        """Calculate IoU (Intersection over Union)"""
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        return intersection / (union + 1e-7)

    def calculate_precision_recall(self, y_true, y_pred):
        """Calculate precision and recall"""
        true_positives = np.sum(y_true * y_pred)
        false_positives = np.sum((1 - y_true) * y_pred)
        false_negatives = np.sum(y_true * (1 - y_pred))
        
        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)
        
        return precision, recall
    
    def display_segmentation_results(self, results):
        """Display segmentation evaluation results with comprehensive charts"""
        if not results['dice']:
            self.eval_results_text.append("No valid results to display.")
            return
        
        # Calculate statistics
        mean_accuracy = np.mean(results['accuracy'])
        mean_iou = np.mean(results['iou'])
        mean_dice = np.mean(results['dice'])
        mean_precision = np.mean(results['precision'])
        mean_recall = np.mean(results['recall'])
        
        std_accuracy = np.std(results['accuracy'])
        std_iou = np.std(results['iou'])
        std_dice = np.std(results['dice'])
        std_precision = np.std(results['precision'])
        std_recall = np.std(results['recall'])
        
        # Display text results
        self.eval_results_text.append(f"\nSegmentation Evaluation Results:")
        self.eval_results_text.append(f"Number of images evaluated: {len(results['dice'])}")
        self.eval_results_text.append(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        self.eval_results_text.append(f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
        self.eval_results_text.append(f"Mean Dice Coefficient: {mean_dice:.4f} ± {std_dice:.4f}")
        self.eval_results_text.append(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
        self.eval_results_text.append(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
        
        # Create box plots for each metric
        fig1 = plt.figure(figsize=(8, 6))
        metrics = ['accuracy', 'iou', 'dice', 'precision', 'recall']
        metric_values = [results[m] for m in metrics]
        plt.boxplot(metric_values, labels=[m.capitalize() for m in metrics])
        plt.title('Distribution of Metrics')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        canvas1 = FigureCanvas(fig1)
        self.eval_chart_layout.addWidget(canvas1)
        
        # Histogram of IoU scores
        fig2 = plt.figure(figsize=(8, 6))
        plt.hist(results['iou'], bins=20, alpha=0.7, color='#3498db', edgecolor='black')
        plt.axvline(mean_iou, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean IoU: {mean_iou:.4f}')
        plt.title('IoU Score Distribution')
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        canvas2 = FigureCanvas(fig2)
        self.eval_chart_layout.addWidget(canvas2)
        
        # Precision-Recall scatter plot
        fig3 = plt.figure(figsize=(8, 6))
        plt.scatter(results['recall'], results['precision'], c=results['iou'], 
                cmap='viridis', alpha=0.7, edgecolors='w', linewidths=0.5)
        plt.colorbar(label='IoU Score')
        plt.title('Precision vs. Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        canvas3 = FigureCanvas(fig3)
        self.eval_chart_layout.addWidget(canvas3)



    def evaluate_classification(self):
        """Evaluate the classification model on the dataset"""
        if not hasattr(self, 'image_files') or not self.image_files:
            QMessageBox.warning(self, "No Images", "Please select a valid dataset folder first.")
            return
        
        # Clear previous results
        self.eval_results_text.clear()
        self.clear_chart_layout()
        
        # Initialize lists for true and predicted labels and additional metrics
        results = {
            'image_name': [],
            'true_label': [],
            'predicted_label': [],
            'confidence': []
        }
        
        # Update UI
        self.eval_results_text.append("Starting classification evaluation...")
        QApplication.processEvents()
        
        # Load the classifier
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            classifier_path = os.path.join(script_dir, "models", "classifier.pkl")
            clf_model = joblib.load(classifier_path)
        except Exception as e:
            self.eval_results_text.append(f"Failed to load classifier: {str(e)}")
            return
        
        # Limit to 100 images
        max_images = min(100, min(len(self.image_files), len(self.mask_files)))
        
        # Process images
        for i in range(max_images):
            # Update progress
            self.eval_progress.setValue((i+1) * 100 // max_images)
            QApplication.processEvents()
            
            # Get ground truth label
            true_label = self.get_ground_truth_label(self.image_files[i])
            
            # Load image and mask
            image_path = os.path.join(self.image_folder, self.image_files[i])
            mask_path = os.path.join(self.mask_folder, self.mask_files[i])
            
            original_image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if original_image is None or mask is None:
                continue
            
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            mask = (mask > 0).astype(np.uint8)
            
            # Extract features and classify
            try:
                features = extract_features(original_image, mask)
                pred_label = clf_model.predict([features])[0]
                
                # Get confidence if available
                confidence = 0
                if hasattr(clf_model, 'predict_proba'):
                    probs = clf_model.predict_proba([features])[0]
                    confidence = np.max(probs)
                
                # Store results
                results['image_name'].append(self.image_files[i])
                results['true_label'].append(true_label)
                results['predicted_label'].append(pred_label)
                results['confidence'].append(confidence)
                
            except Exception as e:
                self.eval_results_text.append(f"Error classifying {self.image_files[i]}: {str(e)}")
        
        # Calculate and display classification results
        self.display_classification_results(results)

    def display_classification_results(self, results):
        """Display classification evaluation results with advanced visualizations"""
        if not results['true_label'] or not results['predicted_label']:
            self.eval_results_text.append("No valid classification results to display.")
            return
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import seaborn as sns
        
        # Convert results to numpy arrays for easier processing
        true_labels = np.array(results['true_label'])
        pred_labels = np.array(results['predicted_label'])
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        report_dict = classification_report(true_labels, pred_labels, output_dict=True)
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Get unique labels (in order)
        unique_labels = sorted(list(set(true_labels) | set(pred_labels)))
        
        # Display text results
        self.eval_results_text.append(f"\nClassification Evaluation Results:")
        self.eval_results_text.append(f"Number of samples evaluated: {len(true_labels)}")
        self.eval_results_text.append(f"Accuracy: {accuracy:.4f}")
        self.eval_results_text.append("\nClassification Report:")
        
        # Format and display classification report
        for label in unique_labels:
            if label in report_dict:
                precision = report_dict[label]['precision']
                recall = report_dict[label]['recall']
                f1 = report_dict[label]['f1-score']
                support = report_dict[label]['support']
                self.eval_results_text.append(f"  {label}:")
                self.eval_results_text.append(f"    Precision: {precision:.4f}")
                self.eval_results_text.append(f"    Recall: {recall:.4f}")
                self.eval_results_text.append(f"    F1-score: {f1:.4f}")
                self.eval_results_text.append(f"    Support: {support}")
        
        # Confusion matrix visualization with improved styling
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111)
        
        # Normalize confusion matrix for better visualization
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels, ax=ax1)
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_title('Confusion Matrix (numbers: counts, colors: normalized)')
        plt.tight_layout()
        
        canvas1 = FigureCanvas(fig1)
        self.eval_chart_layout.addWidget(canvas1)
        
        # Performance metrics by class
        fig2 = plt.figure(figsize=(12, 6))
        
        # Collect metrics per class
        classes = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for label in unique_labels:
            if label in report_dict:
                classes.append(label)
                precisions.append(report_dict[label]['precision'])
                recalls.append(report_dict[label]['recall'])
                f1_scores.append(report_dict[label]['f1-score'])
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precisions, width, label='Precision', color='#3498db')
        plt.bar(x, recalls, width, label='Recall', color='#2ecc71')
        plt.bar(x + width, f1_scores, width, label='F1-score', color='#e74c3c')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Class')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        canvas2 = FigureCanvas(fig2)
        self.eval_chart_layout.addWidget(canvas2)
        
        # Confidence distribution histogram (if available)
        if 'confidence' in results and results['confidence'] and any(c > 0 for c in results['confidence']):
            fig3 = plt.figure(figsize=(10, 6))
            
            # Split confidences by correct and incorrect predictions
            confidences = np.array(results['confidence'])
            correct_pred = (true_labels == pred_labels)
            
            conf_correct = confidences[correct_pred]
            conf_incorrect = confidences[~correct_pred]
            
            plt.hist([conf_correct, conf_incorrect], bins=20, alpha=0.7, 
                    label=['Correct Predictions', 'Incorrect Predictions'],
                    color=['#2ecc71', '#e74c3c'])
            
            plt.axvline(np.mean(confidences), color='navy', linestyle='dashed', 
                    linewidth=2, label=f'Mean Confidence: {np.mean(confidences):.4f}')
            
            plt.title('Prediction Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            canvas3 = FigureCanvas(fig3)
            self.eval_chart_layout.addWidget(canvas3)


    def clear_chart_layout(self):
        """Clear all widgets from the chart layout"""
        for i in reversed(range(self.eval_chart_layout.count())):
            widget = self.eval_chart_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Add placeholder text back
        placeholder_label = QLabel("Charts will appear here after evaluation")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: #888; font-style: italic;")
        self.eval_chart_layout.addWidget(placeholder_label)

    def get_ground_truth_label(self, image_file):
        """Get ground truth label for an image"""
        # In a real application, this would extract labels from filenames or a mapping file
        # For demonstration, return a random cell type
        import random
        return random.choice(['Neutrophil', 'Lymphocyte', 'Monocyte', 'Eosinophil', 'Basophil'])


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