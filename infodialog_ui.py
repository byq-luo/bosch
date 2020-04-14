# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'infodialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_InfoDialog(object):
    def setupUi(self, InfoDialog):
        InfoDialog.setObjectName("InfoDialog")
        InfoDialog.resize(950, 350)
        self.horizontalLayout = QtWidgets.QHBoxLayout(InfoDialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setObjectName("layout")
        self.label_1 = QtWidgets.QLabel(InfoDialog)
        self.label_1.setObjectName("label_1")
        self.layout.addWidget(self.label_1)
        self.label_2 = QtWidgets.QLabel(InfoDialog)
        self.label_2.setObjectName("label_2")
        self.layout.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(InfoDialog)
        self.label_3.setObjectName("label_3")
        self.layout.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(InfoDialog)
        self.label_4.setObjectName("label_4")
        self.layout.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(InfoDialog)
        self.label_5.setObjectName("label_5")
        self.layout.addWidget(self.label_5)
        self.horizontalLayout.addLayout(self.layout)
        self.plotLayout = QtWidgets.QVBoxLayout()
        self.plotLayout.setObjectName("plotLayout")
        self.horizontalLayout.addLayout(self.plotLayout)

        self.retranslateUi(InfoDialog)
        QtCore.QMetaObject.connectSlotsByName(InfoDialog)

    def retranslateUi(self, InfoDialog):
        _translate = QtCore.QCoreApplication.translate
        InfoDialog.setWindowTitle(_translate("InfoDialog", "Info"))
        self.label_1.setText(_translate("InfoDialog", "TextLabel"))
        self.label_2.setText(_translate("InfoDialog", "TextLabel"))
        self.label_3.setText(_translate("InfoDialog", "TextLabel"))
        self.label_4.setText(_translate("InfoDialog", "TextLabel"))
        self.label_5.setText(_translate("InfoDialog", "TextLabel"))
