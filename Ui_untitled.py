# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\research\OCRSys\untitled.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(135, 107)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(30, 10, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.npm)
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 40, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.server)
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 70, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.quit)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "SYS"))
        self.pushButton.setText(_translate("Dialog", "NPM"))
        self.pushButton_2.setText(_translate("Dialog", "Server"))
        self.pushButton_3.setText(_translate("Dialog", "Quit"))

    def quit(self):
        exit(0)
    
    def npm(self):
        os.system("start npm run start")
    
    def server(self):
        os.system("start python ./server/server_sogou.py")

if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Dialog()

    ui.setupUi(MainWindow) 
    MainWindow.show()
    sys.exit(app.exec_()) 