# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'exReID.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1601, 629)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.queryButton = QtWidgets.QPushButton(self.centralwidget)
        self.queryButton.setGeometry(QtCore.QRect(20, 40, 89, 25))
        self.queryButton.setObjectName("queryButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 10, 251, 31))
        self.label.setObjectName("label")
        self.query = QtWidgets.QLabel(self.centralwidget)
        self.query.setGeometry(QtCore.QRect(30, 80, 101, 201))
        self.query.setAutoFillBackground(True)
        self.query.setLineWidth(1)
        self.query.setMidLineWidth(0)
        self.query.setText("")
        self.query.setObjectName("query")
        self.galleryButton = QtWidgets.QPushButton(self.centralwidget)
        self.galleryButton.setGeometry(QtCore.QRect(350, 40, 89, 25))
        self.galleryButton.setObjectName("galleryButton")
        self.galleryLine = QtWidgets.QLineEdit(self.centralwidget)
        self.galleryLine.setGeometry(QtCore.QRect(450, 40, 201, 25))
        self.galleryLine.setObjectName("galleryLine")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(160, 300, 781, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(10, 310, 118, 3))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(140, 80, 16, 201))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.rpta3 = QtWidgets.QLabel(self.centralwidget)
        self.rpta3.setGeometry(QtCore.QRect(430, 80, 101, 201))
        self.rpta3.setAutoFillBackground(True)
        self.rpta3.setLineWidth(1)
        self.rpta3.setMidLineWidth(0)
        self.rpta3.setText("")
        self.rpta3.setObjectName("rpta3")
        self.rpta4 = QtWidgets.QLabel(self.centralwidget)
        self.rpta4.setGeometry(QtCore.QRect(560, 80, 101, 201))
        self.rpta4.setAutoFillBackground(True)
        self.rpta4.setLineWidth(1)
        self.rpta4.setMidLineWidth(0)
        self.rpta4.setText("")
        self.rpta4.setObjectName("rpta4")
        self.rpta5 = QtWidgets.QLabel(self.centralwidget)
        self.rpta5.setGeometry(QtCore.QRect(680, 80, 101, 201))
        self.rpta5.setAutoFillBackground(True)
        self.rpta5.setLineWidth(1)
        self.rpta5.setMidLineWidth(0)
        self.rpta5.setText("")
        self.rpta5.setObjectName("rpta5")
        self.rpta6 = QtWidgets.QLabel(self.centralwidget)
        self.rpta6.setGeometry(QtCore.QRect(810, 80, 101, 201))
        self.rpta6.setAutoFillBackground(True)
        self.rpta6.setLineWidth(1)
        self.rpta6.setMidLineWidth(0)
        self.rpta6.setText("")
        self.rpta6.setObjectName("rpta6")
        self.rptaEx4 = QtWidgets.QLabel(self.centralwidget)
        self.rptaEx4.setGeometry(QtCore.QRect(560, 330, 101, 201))
        self.rptaEx4.setAutoFillBackground(True)
        self.rptaEx4.setLineWidth(1)
        self.rptaEx4.setMidLineWidth(0)
        self.rptaEx4.setText("")
        self.rptaEx4.setObjectName("rptaEx4")
        self.rptaEx6 = QtWidgets.QLabel(self.centralwidget)
        self.rptaEx6.setGeometry(QtCore.QRect(810, 330, 101, 201))
        self.rptaEx6.setAutoFillBackground(True)
        self.rptaEx6.setLineWidth(1)
        self.rptaEx6.setMidLineWidth(0)
        self.rptaEx6.setText("")
        self.rptaEx6.setObjectName("rptaEx6")
        self.rptaEx5 = QtWidgets.QLabel(self.centralwidget)
        self.rptaEx5.setGeometry(QtCore.QRect(680, 330, 101, 201))
        self.rptaEx5.setAutoFillBackground(True)
        self.rptaEx5.setLineWidth(1)
        self.rptaEx5.setMidLineWidth(0)
        self.rptaEx5.setText("")
        self.rptaEx5.setObjectName("rptaEx5")
        self.rptaEx3 = QtWidgets.QLabel(self.centralwidget)
        self.rptaEx3.setGeometry(QtCore.QRect(430, 330, 101, 201))
        self.rptaEx3.setAutoFillBackground(True)
        self.rptaEx3.setLineWidth(1)
        self.rptaEx3.setMidLineWidth(0)
        self.rptaEx3.setText("")
        self.rptaEx3.setObjectName("rptaEx3")
        self.rptaEx1 = QtWidgets.QLabel(self.centralwidget)
        self.rptaEx1.setGeometry(QtCore.QRect(180, 330, 101, 201))
        self.rptaEx1.setAutoFillBackground(True)
        self.rptaEx1.setLineWidth(1)
        self.rptaEx1.setMidLineWidth(0)
        self.rptaEx1.setText("")
        self.rptaEx1.setObjectName("rptaEx1")
        self.rptaEx2 = QtWidgets.QLabel(self.centralwidget)
        self.rptaEx2.setGeometry(QtCore.QRect(300, 330, 101, 201))
        self.rptaEx2.setAutoFillBackground(True)
        self.rptaEx2.setLineWidth(1)
        self.rptaEx2.setMidLineWidth(0)
        self.rptaEx2.setText("")
        self.rptaEx2.setObjectName("rptaEx2")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(140, 310, 16, 281))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(10, 430, 118, 3))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.riseButton = QtWidgets.QPushButton(self.centralwidget)
        self.riseButton.setGeometry(QtCore.QRect(20, 380, 89, 25))
        self.riseButton.setObjectName("riseButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 330, 141, 41))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 450, 111, 31))
        self.label_3.setObjectName("label_3")
        self.xlayersButton = QtWidgets.QPushButton(self.centralwidget)
        self.xlayersButton.setGeometry(QtCore.QRect(10, 520, 89, 25))
        self.xlayersButton.setObjectName("xlayersButton")
        self.layersOptions = QtWidgets.QComboBox(self.centralwidget)
        self.layersOptions.setGeometry(QtCore.QRect(10, 480, 91, 25))
        self.layersOptions.setObjectName("layersOptions")
        self.layersOptions.addItem("")
        self.layersOptions.addItem("")
        self.layersOptions.addItem("")
        self.layersOptions.setItemText(2, "")
        self.queryLine = QtWidgets.QLineEdit(self.centralwidget)
        self.queryLine.setGeometry(QtCore.QRect(120, 40, 211, 25))
        self.queryLine.setObjectName("queryLine")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(50, 290, 61, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(200, 280, 61, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(330, 280, 61, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(450, 280, 61, 21))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(580, 280, 61, 21))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(710, 280, 61, 21))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(840, 280, 61, 21))
        self.label_10.setObjectName("label_10")
        self.cleanButton = QtWidgets.QPushButton(self.centralwidget)
        self.cleanButton.setGeometry(QtCore.QRect(770, 30, 91, 41))
        self.cleanButton.setObjectName("cleanButton")
        self.widgetResult = QtWidgets.QWidget(self.centralwidget)
        self.widgetResult.setGeometry(QtCore.QRect(150, 130, 1451, 451))
        self.widgetResult.setAutoFillBackground(True)
        self.widgetResult.setObjectName("widgetResult")
        self.progressBarMasks = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBarMasks.setGeometry(QtCore.QRect(310, 100, 271, 23))
        self.progressBarMasks.setProperty("value", 24)
        self.progressBarMasks.setObjectName("progressBarMasks")
        self.masksLabel = QtWidgets.QLabel(self.centralwidget)
        self.masksLabel.setGeometry(QtCore.QRect(170, 100, 131, 21))
        self.masksLabel.setObjectName("masksLabel")
        self.explainLabel = QtWidgets.QLabel(self.centralwidget)
        self.explainLabel.setGeometry(QtCore.QRect(640, 100, 131, 21))
        self.explainLabel.setObjectName("explainLabel")
        self.progressBarExplain = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBarExplain.setGeometry(QtCore.QRect(780, 100, 271, 23))
        self.progressBarExplain.setProperty("value", 24)
        self.progressBarExplain.setObjectName("progressBarExplain")
        self.waitLabel = QtWidgets.QLabel(self.centralwidget)
        self.waitLabel.setGeometry(QtCore.QRect(170, 70, 191, 21))
        self.waitLabel.setText("")
        self.waitLabel.setObjectName("waitLabel")
        self.query_2 = QtWidgets.QLabel(self.centralwidget)
        self.query_2.setGeometry(QtCore.QRect(10, 450, 121, 111))
        self.query_2.setAutoFillBackground(True)
        self.query_2.setLineWidth(1)
        self.query_2.setMidLineWidth(0)
        self.query_2.setText("")
        self.query_2.setObjectName("query_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1601, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Explanation of ReID"))
        self.queryButton.setText(_translate("MainWindow", "Select"))
        self.label.setText(_translate("MainWindow", "Select image like your query image"))
        self.galleryButton.setText(_translate("MainWindow", "gallery"))
        self.riseButton.setText(_translate("MainWindow", "Explain"))
        self.label_2.setText(_translate("MainWindow", "Explanation of ReID"))
        self.label_3.setText(_translate("MainWindow", " Layers Explain"))
        self.xlayersButton.setText(_translate("MainWindow", "Explain"))
        self.layersOptions.setItemText(0, _translate("MainWindow", "conv1"))
        self.layersOptions.setItemText(1, _translate("MainWindow", "conv2"))
        self.label_4.setText(_translate("MainWindow", "Query"))
        self.label_5.setText(_translate("MainWindow", "top 1"))
        self.label_6.setText(_translate("MainWindow", "Top 2"))
        self.label_7.setText(_translate("MainWindow", "Top3"))
        self.label_8.setText(_translate("MainWindow", "top4"))
        self.label_9.setText(_translate("MainWindow", "top5"))
        self.label_10.setText(_translate("MainWindow", "top6"))
        self.cleanButton.setText(_translate("MainWindow", "clean"))
        self.masksLabel.setText(_translate("MainWindow", "* Generate masks"))
        self.explainLabel.setText(_translate("MainWindow", "* Explaining"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

