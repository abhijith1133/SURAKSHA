import sys
import cv2
import time
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer


class WelcomePage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Car image label (you'll need to add the actual image)
        self.car_label = QLabel()
        self.car_label.setAlignment(Qt.AlignCenter)

        # Welcome text
        welcome_label = QLabel("WELCOME TO\nDRIVER FAULT DETECTION")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setFont(QFont('Arial', 24, QFont.Bold))
        welcome_label.setWordWrap(True)

        # Safety message
        safety_label = QLabel("LET'S BE SAFE ON ROADS AND LET\nGET ALERT WHEN WE GO SLEEP")
        safety_label.setAlignment(Qt.AlignCenter)
        safety_label.setFont(QFont('Arial', 12))
        safety_label.setWordWrap(True)

        # Get Started button
        start_button = QPushButton("Get Started")
        start_button.setFixedSize(200, 50)
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #FFD700;
                border-radius: 25px;
                font-size: 18px;
                font-weight: bold;      
            }
            QPushButton:hover {
                background-color: #FFC700;
            }
        """)
        start_button.clicked.connect(self.start_monitoring)

        # Add widgets to layout
        layout.addWidget(self.car_label)
        layout.addWidget(welcome_label)
        layout.addWidget(safety_label)
        layout.addWidget(start_button)
        layout.setAlignment(Qt.AlignCenter)

        self.setLayout(layout)

    def start_monitoring(self):
        self.stacked_widget.setCurrentIndex(1)


class MonitoringPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()
        self.setup_face_detection()

    def initUI(self):
        layout = QVBoxLayout()

        # Back button
        back_button = QPushButton("←")
        back_button.setFixedSize(40, 40)
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        # Title
        title_label = QLabel("Face Scanning")
        title_label.setFont(QFont('Arial', 20, QFont.Bold))

        # Video feed
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #D3D3D3;")

        # Driver status
        self.status_label = QLabel("DRIVER STATUS:")
        self.status_label.setFont(QFont('Arial', 14, QFont.Bold))

        # Warning label
        self.warning_label = QLabel("Status: Monitoring...")
        self.warning_label.setFont(QFont('Arial', 14))
        self.warning_label.setStyleSheet("color: green;")

        # Add widgets to layout
        layout.addWidget(back_button)
        layout.addWidget(title_label)
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.status_label)
        layout.addWidget(self.warning_label)

        self.setLayout(layout)

    def setup_face_detection(self):
        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Initialize variables
        self.eye_closed_start_time = None
        self.eye_closed_threshold = 4
        self.eye_detected_last_time = None

        # Start video capture
        self.cap = cv2.VideoCapture(0)

        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        eyes_detected = False
        current_time = time.time()

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15))

            if len(eyes) > 0:
                eyes_detected = True
                self.eye_detected_last_time = current_time

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Fatigue detection logic
        if eyes_detected:
            self.eye_closed_start_time = None
            self.warning_label.setText("Status: Normal")
            self.warning_label.setStyleSheet("color: green;")
        else:
            if self.eye_detected_last_time and (current_time - self.eye_detected_last_time < 1):
                return

            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time

            elapsed_time = current_time - self.eye_closed_start_time if self.eye_closed_start_time else 0

            if elapsed_time >= self.eye_closed_threshold:
                self.warning_label.setText("WARNING: Driver is sleeping!")
                self.warning_label.setStyleSheet("color: red;")

        # Convert frame for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Driver Fault Detection System')
        self.setGeometry(100, 100, 800, 600)

        # Create stacked widget to handle page switching
        self.stacked_widget = QStackedWidget()

        # Create pages
        welcome_page = WelcomePage(self.stacked_widget)
        monitoring_page = MonitoringPage(self.stacked_widget)

        # Add pages to stacked widget
        self.stacked_widget.addWidget(welcome_page)
        self.stacked_widget.addWidget(monitoring_page)

        # Set central widget
        self.setCentralWidget(self.stacked_widget)


def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()