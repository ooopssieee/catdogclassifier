import sys
from app import CatDogClassifierApp
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QPushButton, 
    QFileDialog, QWidget, QFrame, QHBoxLayout
)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CatDogClassifierApp()
    window.show()
    sys.exit(app.exec_()) 
