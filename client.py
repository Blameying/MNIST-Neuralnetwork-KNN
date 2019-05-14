import sys
from PyQt5.QtCore import (QObject,pyqtSignal,QThread)
from PyQt5.QtGui import (QPixmap,QPalette,QColor,QPainter,QPen)
from PyQt5.QtWidgets import (QApplication,QWidget,QHBoxLayout,QVBoxLayout,QLabel,QPushButton,QLineEdit,QComboBox)
from neuralnetwork import *
from knn_mnist import *
from mnistLoader import *
import time
import numpy as np


class KNNThread(QThread):
    update = pyqtSignal(str)
    finishSignal = pyqtSignal()
    def __init__(self,test_images,test_labels,images,labels,k):
        super(KNNThread,self).__init__()
        self.test_images = test_images
        self.test_labels = test_labels
        self.images = images
        self.labels = labels
        self.k = k
    def run(self):
        msg = ""
        right = 0
        wrong = 0
        time_start = time.time()
        for index in range(len(self.test_images)):
            msg = "正在检测: "+str(index)+" times"
            self.update.emit(msg)
            result = knn(self.test_images[index],self.images,self.labels,self.k)
            if(result==self.test_labels[index]):
                right+=1
            else:
                wrong+=1
        time_end = time.time()
        msg = "检测完毕,\n识别率:"+str(right/(right+wrong))+"\n耗时:"+str(time_end-time_start)
        self.update.emit(msg)
        self.finishSignal.emit()

class NeuralNetworkThread(QThread):
    update = pyqtSignal(str)
    finishSignal = pyqtSignal()
    def __init__(self,test_images,test_labels,images,labels,network):
        super(NeuralNetworkThread,self).__init__()
        self.test_images = test_images
        self.test_labels = test_labels
        self.images = images
        self.labels = labels
        self.network = network 

    def setText(self,msg):
        self.update.emit(msg)

    def run(self):
        msg = ""
        right = 0
        wrong = 0
        out = getOutputFormat(self.labels)
        src = getZipImagesAndTables(self.images,out)
        time_start = time.time()
        self.network.train(src,10,10,self)
        for index in range(len(self.test_images)):
            msg = "正在检测: "+str(index)+" times"
            self.update.emit(msg)
            result = self.network.test(self.test_images[index])
            if(result==self.test_labels[index]):
                right+=1
            else:
                wrong+=1
        time_end = time.time()
        msg = "检测完毕,\n识别率:"+str(right/(right+wrong))+"\n耗时:"+str(time_end-time_start)
        self.update.emit(msg)
        self.finishSignal.emit()
        self.network.saveToFile()
        
class DrawingBoard(QWidget):
    def __init__(self):
        super(DrawingBoard,self).__init__()
        self.buffer = np.zeros((280,280),dtype=np.uint8)
        self.pixmap = QPixmap(280,280)
        self.pixmap.fill(QColor(222,222,222))
        self.painter = QPainter(self.pixmap)
        self.painter.setPen(QPen(QColor(0,0,0),10))
        self.label = QLabel()
        self.label.setPixmap(self.pixmap)
        self.if_mouse_press = False 
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setFixedSize(280,280)

    def clear(self):
        self.pixmap.fill(QColor(222,222,222))
        self.buffer = np.zeros((280,280),dtype=np.uint8)
        self.label.setPixmap(self.pixmap)

    def mouseMoveEvent(self,e):
        x=e.x()
        y=e.y()
        if self.if_mouse_press:
            if x>=0 and x<280 and y>=0 and y<280:
                self.painter.drawPoint(e.x(),e.y())
                self.buffer[e.y()-8:e.y()+8,e.x()-8:e.x()+8]=1
                self.label.setPixmap(self.pixmap)
    def mousePressEvent(self,e):
        self.if_mouse_press = True

    def mouseReleaseEvent(self,e):  
        self.if_mouse_press = False

    def mouseDoubleClickEvent(self,e):
        self.clear()

    def getData(self):
        return self.buffer[14:266:9,14:266:9]

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.algorithms = ['KNN','Neural-Network']
        self.images,self.labels = load_data('./')
        self.test_images,self.test_labels = load_data('./',kind='t10k')
        self.network = NNetWork(784,40,10,4)
        self.ui();

    def ui(self):
        self.setWindowTitle("手写数字识别")
        self.setFixedSize(600,300)
        self.mainLayout = QHBoxLayout()
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()
        self.drawingBoard = DrawingBoard()
        self.leftLayout.addWidget(self.drawingBoard)

        label = QLabel("算法")
        self.combobox=QComboBox()
        self.combobox.addItems(self.algorithms)
        self.settingButton=QPushButton("设置")
        algorithmsChooseToolLayout = QHBoxLayout()
        algorithmsChooseToolLayout.addWidget(label)
        algorithmsChooseToolLayout.addWidget(self.combobox)
        algorithmsChooseToolLayout.addWidget(self.settingButton)

        self.currentChoice = QLabel("未选择")

        self.trainButton = QPushButton("训练")
        self.testButton = QPushButton("测试")
        trainOrTestLayout = QHBoxLayout()
        trainOrTestLayout.addWidget(self.trainButton)
        trainOrTestLayout.addWidget(self.testButton)

        self.infoDisplay = QLabel("无信息")

        self.rightLayout.addLayout(algorithmsChooseToolLayout)
        self.rightLayout.addWidget(self.currentChoice)
        self.rightLayout.addLayout(trainOrTestLayout)
        self.rightLayout.addWidget(self.infoDisplay)

        self.mainLayout.addLayout(self.leftLayout)
        self.mainLayout.addLayout(self.rightLayout)
        self.setLayout(self.mainLayout)
    
        self.settingButton.clicked.connect(self.settingButtonClicked)
        self.trainButton.clicked.connect(self.trainTask)
        self.testButton.clicked.connect(self.testTask)

    def settingButtonClicked(self):
        self.currentChoice.setText(self.combobox.currentText())
        
    def updateInfoDisplay(self,msg):
        self.infoDisplay.setText(msg)

    def trainFinish(self):
        self.trainButton.setEnabled(True)
        self.testButton.setEnabled(True)

    def trainTask(self):
        text = self.currentChoice.text()
        if(text==self.algorithms[0]):
            self.trainButton.setEnabled(False)
            self.testButton.setEnabled(False)
            self.knnThread = KNNThread(self.test_images,self.test_labels,self.images,self.labels,15)
            self.knnThread.update.connect(self.updateInfoDisplay)
            self.knnThread.finishSignal.connect(self.trainFinish)
            self.knnThread.start()
        elif(text==self.algorithms[1]):
            self.trainButton.setEnabled(False)
            self.testButton.setEnabled(False)
            self.neuralNetworkThread = NeuralNetworkThread(self.test_images,self.test_labels,self.images,self.labels,self.network)
            self.neuralNetworkThread.update.connect(self.updateInfoDisplay)
            self.neuralNetworkThread.finishSignal.connect(self.trainFinish)
            self.neuralNetworkThread.start()
        else:
            self.infoDisplay.setText("未选择任何算法")
        
    def testTask(self):
        text = self.currentChoice.text()
        if(text==self.algorithms[0]):
            data = self.drawingBoard.getData().reshape(784,)
            self.trainButton.setEnabled(False)
            self.testButton.setEnabled(False)
            result = knn(data,self.images,self.labels,20)
            self.infoDisplay.setText("KNN 结果:\n"+str(result))
            self.trainFinish()
        elif(text==self.algorithms[1]):
            data = self.drawingBoard.getData().reshape(784,)
            self.trainButton.setEnabled(False)
            self.testButton.setEnabled(False)
            self.network.loadFromfile()
            result = self.network.test(data)
            self.infoDisplay.setText("NeuralNetwork结果:\n"+str(result))
            self.trainFinish()
        else:
            self.infoDisplay.setText("未选择任何算法")
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

