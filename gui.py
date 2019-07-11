# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\research\OCRSys\untitled.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import subprocess
import _thread

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(20, 10, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.hello)
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setEnabled(False)
        self.textBrowser.setGeometry(QtCore.QRect(20, 40, 351, 101))
        self.textBrowser.setObjectName("textBrowser")
        

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "PushButton"))

    

    def hello(self):
        def printpipe(pipe):
            pass
            while pipe.read().decode():
                print(pipe.read().decode())
            
        print("hi")
        pipe = subprocess.Popen("python server_sogou.py", shell = True, stdout = subprocess.PIPE).stdout
        #self.textBrowser.setText(str(pipe.read()))
        _thread.start_new_thread( printpipe, (pipe, ) )

        

if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Dialog()

    ui.setupUi(MainWindow) 
    MainWindow.show()
    sys.exit(app.exec_()) 