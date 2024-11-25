from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QPushButton, 
    QFileDialog, QWidget, QFrame, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QTimer
from torchvision import transforms
from PIL import Image
import cv2
from inference import *


class CatDogClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.is_preview_active = False

    def initUI(self):
        self.setWindowTitle("Cat vs Dog Classifier")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QPushButton {
                background-color: #2d2d2d;
                border: none;
                padding: 8px;
                color: #4a9eff;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:disabled {
                color: #666666;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title_bar = QFrame()
        title_bar.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-bottom: 1px solid #333333;
            }
        """)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 5, 10, 5)
        
        self.preview_button = QPushButton("▶ Live Preview")
        self.preview_button.setStyleSheet("""
            QPushButton {
                color: #4a9eff;
                font-size: 14px;
                text-align: left;
                padding: 8px 15px;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
        """)
        self.preview_button.clicked.connect(self.toggle_preview)
        title_layout.addWidget(self.preview_button, alignment=Qt.AlignLeft)
        title_layout.addStretch()

        layout.addWidget(title_bar)
        
        self.drop_area = DropArea()
        layout.addWidget(self.drop_area)

        bottom_bar = QFrame()
        bottom_bar.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-top: 1px solid #333333;
            }
        """)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(10, 5, 10, 5)

        left_button_layout = QHBoxLayout()
        
        add_button = QPushButton("+")
        add_button.setStyleSheet("QPushButton { font-size: 20px; padding: 5px 15px; }")
        add_button.clicked.connect(self.open_file_dialog)
        
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_image)

        left_button_layout.addWidget(add_button)
        left_button_layout.addWidget(clear_button)
        left_button_layout.addStretch()
        
        bottom_layout.addLayout(left_button_layout)
        
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.result_label.setStyleSheet("color: #4a9eff; font-size: 16px;")
        bottom_layout.addWidget(self.result_label)

        layout.addWidget(bottom_bar)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = None

    def toggle_preview(self):
        if not self.is_preview_active:
            self.start_preview()
        else:
            self.stop_preview()

    def start_preview(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.result_label.setText("Error: Could not access the camera.")
            return

        self.timer.start(30)
        self.is_preview_active = True
        self.preview_button.setText("■ Stop Preview")
        self.drop_area.clearImage()

    def stop_preview(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_preview_active = False
        self.preview_button.setText("▶ Live Preview")
        self.drop_area.clearImage()
        self.result_label.setText("")

    def open_file_dialog(self):
        if self.is_preview_active:
            self.stop_preview()
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image File", 
            "", 
            "Images (*.png *.jpg *.jpeg)",
            options=options
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        if self.is_preview_active:
            self.stop_preview()
            
        pixmap = QPixmap(file_path)
        self.drop_area.setImage(pixmap)

        image = Image.open(file_path).convert('RGB')
        prediction = self.predict(image)
        self.result_label.setText(f"Prediction: {prediction}")

    def clear_image(self):
        if self.is_preview_active:
            self.stop_preview()
        self.drop_area.clearImage()
        self.result_label.setText("")

    def predict(self, image):
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            if confidence.item() < 0.85 and confidence.item() > 0.15    :
                return "Other"
            return "Cat" if predicted.item() == 0 else "Dog"


    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            prediction = self.predict(frame_image)
            self.result_label.setText(f"Prediction: {prediction}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = frame.shape
            bytes_per_line = 3 * width
            qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            self.drop_area.setImage(pixmap)

class ImagePlaceholder(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setSvg('''
            <svg width="64" height="64" viewBox="0 0 64 64" fill="#3c3c3c">
                <path d="M55 8H9a1 1 0 00-1 1v46a1 1 0 001 1h46a1 1 0 001-1V9a1 1 0 00-1-1zM10 54V10h44v44H10z"/>
                <path d="M14 44.5l12-12 8 8 12-12 4 4v12H14v-0.5z"/>
                <circle cx="24" cy="24" r="4"/>
            </svg>
        ''')
        self.setStyleSheet("QLabel { background-color: transparent; }")

    def setSvg(self, svg_str):
        pixmap = QPixmap()
        pixmap.loadFromData(svg_str.encode('utf-8'), 'SVG')
        self.setPixmap(pixmap)


class DropArea(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        self.placeholder = ImagePlaceholder()
        self.text_label = QLabel("Drag or Add Images")
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("color: #808080; font-size: 14px;")
        
        layout.addStretch()
        layout.addWidget(self.placeholder, alignment=Qt.AlignCenter)
        layout.addWidget(self.text_label, alignment=Qt.AlignCenter)
        layout.addStretch()
        
        self.setStyleSheet("""
            DropArea {
                background-color: #1e1e1e;
                border: 1px dashed #424242;
                border-radius: 4px;
            }
        """)
        self.setAcceptDrops(True)
        self.setFixedSize(640, 480)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasImage or event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                app = self.parent() 
                if hasattr(app, "load_image"):
                    app.load_image(url.toLocalFile())
                    event.accept()
                else:
                    print("The parent does not have a load_image method.")
                    event.ignore()
            else:
                print("Dropped file is not a local file.")
                event.ignore()
        else:
            print("Dropped data is not a valid URL.")
            event.ignore()



    def setImage(self, pixmap):
        self.placeholder.hide()
        self.text_label.hide()
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.show()

    def clearImage(self):
        self.image_label.clear()
        self.image_label.hide()
        self.placeholder.show()
        self.text_label.show()
