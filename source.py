import PyQt6
import pygame
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.uic import loadUi
import sys
import cv2 as cv
from ultralytics import YOLO

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("GUI/GUIfile.ui",self)
        self.setWindowTitle("Real-time Fire Detection")

        self.cap = None
        self.pushButton.clicked.connect(self.state_cam)
        self.is_playing = False

        self.radioCPU.setChecked(True)
        self.radioCPU.toggled.connect(self.cpu_mode)
        self.radioGPU.toggled.connect(self.gpu_mode)

        self.checkAlarm.stateChanged.connect(self.alarm_mode)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        pygame.mixer.init()
        self.sound = pygame.mixer.Sound("resources/sound.mp3")
        self.useSound = False

        self.model = YOLO("bestFire.pt")
        self.model.to("cpu")


    def state_cam(self):
        if self.is_playing:
            self.timer.stop()
            self.cap.release()
            self.cap=None
            self.pushButton.setText("Start")
            self.screen.setPixmap(QtGui.QPixmap("resources/wait.png"))
        else:
            self.cap = cv.VideoCapture(0)
            self.timer.start(30)
            self.pushButton.setText("Stop")
        self.is_playing = not self.is_playing

    def cpu_mode(self):
        if self.radioCPU.isChecked():
            self.model.to("cpu")
            print("Using cpu")

    def gpu_mode(self):
        try:
            if self.radioGPU.isChecked():
                self.model.to("cuda")
                print("Using cuda")
        except:
            self.show_popup()
            self.radioCPU.setChecked(True)

    def show_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText("GPU is not availabel, use CPU instead!")
        msg.setIcon(QMessageBox.Icon.Critical)

        msg.exec()

    def alarm_mode(self):
        if self.checkAlarm.isChecked():
            self.useSound = True
            print("Use sound")
        else:
            self.useSound = False
            print("No use sound")

    def update_frame(self):
        if self.cap is None:
            return
        ret, first_frame = self.cap.read()
        if ret:
            results =  self.model.predict(source=first_frame)
            frame = results[0].plot()
            fire = results[0].boxes.cls.cpu().numpy().astype('int')
            if fire.size != 0 and self.useSound:
                self.sound.play()
            else: self.sound.stop()

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            h,w,ch = rgb_frame.shape
            bytes_per_line = ch*w
            qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            real_frame =  QPixmap.fromImage(qt_img)

            self.screen.setPixmap(real_frame)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GUI()
    win.show()
    sys.exit(app.exec())