from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import pandas as pd
import cv2
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show
import time
import math
import argparse
import os
from glob import glob
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog, QMainWindow, QPushButton
import source
import pe

global filename1
timestr = time.strftime("%Y %m %d-%H %M")
filename1 = "ouput" + timestr;



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(595, 447)
        MainWindow.setStyleSheet("background-color: rgb(45, 45, 45);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")


        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(370, 70, 171, 31))
        self.pushButton.setStyleSheet("background-color:rgb(35, 35, 35);\n"
"color:white;\n"
"border-style:outset;\n"
"border-radius:10px;\n"
"border-width:1px;\n"
"border-color:white;\n"
"font: bold 14px;\n"
"padding:6px;\n"
"min-width:10px;\n"
)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.facial_detection)
        



        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(370, 160, 171, 31))
        self.pushButton_2.setStyleSheet("background-color:rgb(35, 35, 35);\n"
"color:white;\n"
"border-style:outset;\n"
"border-radius:10px;\n"
"border-width:1px;\n"
"border-color:white;\n"
"font: bold 14px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.graph1)



        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(370, 250, 171, 31))
        self.pushButton_3.setStyleSheet("background-color:rgb(35, 35, 35);\n"
"color:white;\n"
"border-style:outset;\n"
"border-radius:10px;\n"
"border-width:1px;\n"
"border-color:white;\n"
"font: bold 14px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.graph2)




        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(110, 340, 151, 31))
        self.pushButton_4.setStyleSheet("background-color:rgb(35, 35, 35);\n"
"color:white;\n"
"border-style:outset;\n"
"border-radius:10px;\n"
"border-width:1px;\n"
"border-color:white;\n"
"font: bold 14px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.launch_script)



        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(370, 340, 171, 31))
        self.pushButton_5.setStyleSheet("background-color:rgb(35, 35, 35);\n"
"color:white;\n"
"border-style:outset;\n"
"border-radius:10px;\n"
"border-width:1px;\n"
"border-color:white;\n"
"font: bold 14px;\n"
"padding:6px;\n"
"min-width:10px\n;")
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.showDialog)



        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 100, 181, 181))
        self.label.setStyleSheet("background-image:url(:/newPrefix/z.png)")
        self.label.setText("")
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def launch_script(self):
        self.panel = pe.Sheet()
        self.panel.show()

    def showDialog(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setStyleSheet('QMessageBox {color: white;font : bold 24px}')
        msgBox.setText("Your result is saved as %s.csv" %filename1)
        msgBox.setWindowTitle("File name")
        returnValue = msgBox.exec()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FER 2021(PCE)"))
        MainWindow.setWindowIcon(QtGui.QIcon("icon.ico"))
        self.pushButton.setText(_translate("MainWindow", "START"))
        self.pushButton_2.setText(_translate("MainWindow", "PIE GRAPH"))
        self.pushButton_3.setText(_translate("MainWindow", "LINE GRAPH"))
        self.pushButton_4.setText(_translate("MainWindow", "HISTORY"))
        self.pushButton_5.setText(_translate("MainWindow", "FILE NAME"))

    
    def facial_detection(self):
        face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') #calling of haarcascade file
        classifier =load_model('./Emotion_Detection.h5') #classifier

        class_labels = ['Angry','Happy','Neutral','Sad','Surprise'] #emotions in list

        cap = cv2.VideoCapture(0)

        f = open(filename1 + '.txt', 'w') #text file is created

        rs=0

        t=time.time()

        x = []  #gor bar graph
        y = []

        while True:

            ret, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,225,0),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)                  #converting image into array
                    roi = np.expand_dims(roi,axis=0)


                    preds = classifier.predict(roi)[0]
                    print("\nprediction = ",preds)
                    label=class_labels[preds.argmax()]
                    print("\nprediction max = ",preds.argmax())
                    label1 = "{}: {:.2f}%".format(label, max(preds) * 100)
                    print("\nlabel = ",label)
                    ti = time.time() - t
                    print("\nTime=",ti )
                    f.write(str(rs))
                    f.write(",")
                    f.write(label)
                    f.write(",")
                    f.write(str(preds.argmax()))
                    f.write(",")
                    f.write(str(ti))
                    f.write("\n")
                    rs+=1
                    label_position = (x,y)
                    cv2.putText(frame,label1,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                else:
                    cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                #print("\n\n")
            cv2.imshow('FACIAL EMOTION DETECTION(BE IT PROJECT)',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        f.close()#define function for detecting facial expression
        

        

        

    def graph1(self):
        data_files = sorted(glob('ouput2021 04 0*.csv'))
        data = pd.concat((pd.read_csv(file).assign(filename = file) for file in data_files), ignore_index = True)
        f=data['Emotion'].value_counts(normalize=True) * 100
        try:
            k= f[0]
        except IndexError:
            k = 0
        try:
            p1= f[1]
        except IndexError:
            p1 = 0
        try:
            n1= f[2]
        except IndexError:
            n1 = 0    
        try:
            n2= f[3]
        except IndexError:
            n2= 0
        try:
            n3= f[4]
        except IndexError:
            n3 = 0
        K=k
        N=n1+n2+n3
        P=p1;
        data_columns = ['Sr.no','Emotion','Code','Time']
        read_file = pd.read_csv ("%s.txt" %filename1,names=data_columns)
        read_file.to_csv ('%s.csv' %filename1, index=None)
        df = pd.read_csv ("%s.csv" %filename1);
        sums = df.groupby(df["Emotion"])["Sr.no"].nunique()
        plt.title('PIE GRAPH OF EMOTION DETECTED')
        plt.axis('equal');
        plt.pie(sums, labels=sums.index,autopct='%.2f%%');
        if (K>= 44):
            plt.text(0,1.2,"Composure performance is medium",horizontalalignment='center',verticalalignment='top')
        elif(N >= 20 ):
            plt.text(0,1.2,"Composure performance is low",horizontalalignment='center',verticalalignment='top')
        else:
            plt.text(0,1.2,"Composure performance is high",horizontalalignment='center',verticalalignment='top')
            plt.text(0,1.1,"CONFIDENCE IS HIGH",horizontalalignment='center',verticalalignment='top')
            
        plt.show()#pie graph


    def graph2(self):
        data_files = sorted(glob('ouput2021 04 0*.csv'))
        data = pd.concat((pd.read_csv(file).assign(filename = file) for file in data_files), ignore_index = True)
        f=data['Emotion'].value_counts(normalize=True) * 100
        data_columns = ['Sr.no','Emotion','Code','Time']
        read_file = pd.read_csv ("%s.txt" %filename1,names=data_columns)
        read_file.to_csv ('%s.csv' %filename1, index=None)
        df = pd.read_csv ("%s.csv" %filename1);
        plt.title('LINE GRAPH OF EMOTION DETECTED')
        x = df['Time']
        y = df['Emotion']
        plt.plot(x,y)
        plt.ylabel("Emotions")
        plt.show()#simple graph


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())