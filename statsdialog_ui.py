# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'statsdialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_StatsDialog(object):
    def setupUi(self, StatsDialog):
        StatsDialog.setObjectName("StatsDialog")
        StatsDialog.resize(950, 350)
        self.horizontalLayout = QtWidgets.QHBoxLayout(StatsDialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.statsLayout = QtWidgets.QVBoxLayout()
        self.statsLayout.setObjectName("statsLayout")
        self.label_1 = QtWidgets.QLabel(StatsDialog)
        self.label_1.setObjectName("label_1")
        self.statsLayout.addWidget(self.label_1)
        self.label_2 = QtWidgets.QLabel(StatsDialog)
        self.label_2.setObjectName("label_2")
        self.statsLayout.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(StatsDialog)
        self.label_3.setObjectName("label_3")
        self.statsLayout.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(StatsDialog)
        self.label_4.setObjectName("label_4")
        self.statsLayout.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(StatsDialog)
        self.label_5.setObjectName("label_5")
        self.statsLayout.addWidget(self.label_5)
        self.horizontalLayout.addLayout(self.statsLayout)
        self.plotLayout = QtWidgets.QVBoxLayout()
        self.plotLayout.setObjectName("plotLayout")
        self.horizontalLayout.addLayout(self.plotLayout)

        self.retranslateUi(StatsDialog)
        QtCore.QMetaObject.connectSlotsByName(StatsDialog)

    def retranslateUi(self, StatsDialog):
        _translate = QtCore.QCoreApplication.translate
        StatsDialog.setWindowTitle(_translate("StatsDialog", "Info"))
        self.label_1.setText(_translate("StatsDialog", "TextLabel"))
        self.label_2.setText(_translate("StatsDialog", "TextLabel"))
        self.label_3.setText(_translate("StatsDialog", "TextLabel"))
        self.label_4.setText(_translate("StatsDialog", "TextLabel"))
        self.label_5.setText(_translate("StatsDialog", "TextLabel"))
