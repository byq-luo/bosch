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
        self.label_8 = QtWidgets.QLabel(StatsDialog)
        self.label_8.setObjectName("label_8")
        self.statsLayout.addWidget(self.label_8)
        self.label_3 = QtWidgets.QLabel(StatsDialog)
        self.label_3.setObjectName("label_3")
        self.statsLayout.addWidget(self.label_3)
        self.label_6 = QtWidgets.QLabel(StatsDialog)
        self.label_6.setObjectName("label_6")
        self.statsLayout.addWidget(self.label_6)
        self.label_5 = QtWidgets.QLabel(StatsDialog)
        self.label_5.setObjectName("label_5")
        self.statsLayout.addWidget(self.label_5)
        self.label_2 = QtWidgets.QLabel(StatsDialog)
        self.label_2.setObjectName("label_2")
        self.statsLayout.addWidget(self.label_2)
        self.label_7 = QtWidgets.QLabel(StatsDialog)
        self.label_7.setObjectName("label_7")
        self.statsLayout.addWidget(self.label_7)
        self.label_4 = QtWidgets.QLabel(StatsDialog)
        self.label_4.setObjectName("label_4")
        self.statsLayout.addWidget(self.label_4)
        self.horizontalLayout.addLayout(self.statsLayout)
        self.plotLayout = QtWidgets.QVBoxLayout()
        self.plotLayout.setObjectName("plotLayout")
        self.horizontalLayout.addLayout(self.plotLayout)

        self.retranslateUi(StatsDialog)
        QtCore.QMetaObject.connectSlotsByName(StatsDialog)

    def retranslateUi(self, StatsDialog):
        _translate = QtCore.QCoreApplication.translate
        StatsDialog.setWindowTitle(_translate("StatsDialog", "Statistics"))
        self.label_8.setText(_translate("StatsDialog", "<html><head/><body><p><span style=\" font-weight:600;\">Estimated hours saved: </span>42</p></body></html>"))
        self.label_3.setText(_translate("StatsDialog", "<html><head/><body><p><span style=\" font-weight:600;\">Hours of video processed: </span>32</p></body></html>"))
        self.label_6.setText(_translate("StatsDialog", "<html><head/><body><p><span style=\" font-weight:600;\">Processing hours remaining: </span>50</p></body></html>"))
        self.label_5.setText(_translate("StatsDialog", "<html><head/><body><p><span style=\" font-weight:600;\">Number of labels predicted: </span>130</p></body></html>"))
        self.label_2.setText(_translate("StatsDialog", "<html><head/><body><p><span style=\" font-weight:600;\">Number of videos processed: </span>100</p></body></html>"))
        self.label_7.setText(_translate("StatsDialog", "<html><head/><body><p><span style=\" font-weight:600;\">Average video length: </span>5m</p></body></html>"))
        self.label_4.setText(_translate("StatsDialog", "<html><head/><body><p><span style=\" font-weight:600;\">Gpu device name: </span>NVIDIA GTX 1080 Ti</p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    StatsDialog = QtWidgets.QDialog()
    ui = Ui_StatsDialog()
    ui.setupUi(StatsDialog)
    StatsDialog.show()
    sys.exit(app.exec_())
