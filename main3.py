import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class ColorHandDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Tạo giao diện
        self.setWindowTitle("Color and Hand Detection")
        self.camera_label = QLabel(self)
        self.red_label = QLabel("0", self)
        self.blue_label = QLabel("0", self)
        self.yellow_label = QLabel("0", self)

        # Cài đặt layout
        self.init_ui()

        # Camera và bộ định nghĩa màu
        self.cap = cv2.VideoCapture(0)
        self.color_ranges = self.define_color_ranges()

        # Timer để cập nhật khung hình
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # MediaPipe Hands
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def init_ui(self):
        # Bố trí phần camera
        vbox = QVBoxLayout()
        vbox.addWidget(self.camera_label)

        # Bố trí số lượng
        hbox = QHBoxLayout()

        # Nhãn "Số lượng"
        label = QLabel("Số lượng:", self)
        label.setStyleSheet("font-size: 18px;")  # Tăng kích thước font chữ của "Số lượng"
        hbox.addWidget(label)

        # Số lượng đỏ
        self.red_label.setStyleSheet("background-color: red; color: white; font-size: 18px;")  # Tăng kích cỡ font chữ
        hbox.addWidget(self.red_label)

        # Số lượng xanh dương
        self.blue_label.setStyleSheet("background-color: blue; color: white; font-size: 18px;")  # Tăng kích cỡ font chữ
        hbox.addWidget(self.blue_label)

        # Số lượng vàng
        self.yellow_label.setStyleSheet("background-color: yellow; color: black; font-size: 18px;")  # Tăng kích cỡ font chữ
        hbox.addWidget(self.yellow_label)

        vbox.addLayout(hbox)
        self.setLayout(vbox)

    def define_color_ranges(self):
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
        }
        return {k: (np.array(v[0], dtype="uint8"), np.array(v[1], dtype="uint8")) for k, v in color_ranges.items()}

    def detect_colors(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_counts = {'red': 0, 'blue': 0, 'yellow': 0}
        
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Lọc các khối nhỏ
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    color_counts[color_name] += 1
                    print(color_name)  # In tên màu ra terminal

        # Cập nhật số lượng màu
        self.red_label.setText(str(color_counts['red']))
        self.blue_label.setText(str(color_counts['blue']))
        self.yellow_label.setText(str(color_counts['yellow']))

    def detect_hand_mediapipe(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            self.close()  # Đóng chương trình khi phát hiện bàn tay
            print("Phát hiện bàn tay, dừng chương trình.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Phát hiện màu
            self.detect_colors(frame)

            # Phát hiện bàn tay
            self.detect_hand_mediapipe(frame)

            # Hiển thị khung hình
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.cap.release()
        self.hands.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ColorHandDetectionApp()
    window.show()
    sys.exit(app.exec_())
