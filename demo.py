import sys
import os
from decimal import Decimal
import glob
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QProgressBar
from PyQt5.QtGui import QPixmap
import cv2
#from PyQt5.QtGui import QIcon
import tensorflow as tf
#from matplotlib import pyplot as plt
#qtCreatorFile = "exReID.ui" # Enter file here.
#Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
import numpy as np
from explainReID import ReID_Model
import explainReID as xreid

from xReid_ui import Ui_MainWindow
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
#from pReID.personReID2 import personReIdentifier

batch_size = 250
#model_batch = ReID_Model(batch_size)
#model_batch = ReID_Model(1)
#reid = personReIdentifier()

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=10, height=10, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #fig = plt.figure(figsize=(12,12))
        #self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        #variavels para a explicação
        self.model_batch = ReID_Model(batch_size)
        self.pathQuery = None
        self.pathGallery = None
        self.saliency_result = []
        self.layout = QtWidgets.QVBoxLayout()
        #progress bar
        #self.progressBar.setAutoFillBackground
        self.progressBarMasks.setValue(0)
        self.progressBarExplain.setValue(0)
        #self.progress = QProgressBar(self)
        #self.progress.setGeometry(0, 0, 300, 25)
        #self.progress.setMaximum(100)

        #methods of GUI
        
        #self.calc_tax_button.clicked.connect(self.calculate_tax)
        self.queryButton.clicked.connect(self.openQueryNameDialog)
        self.galleryButton.clicked.connect(self.openGalleryNameDialog)
        self.cleanButton.clicked.connect(self.clean_lines)
        self.riseButton.clicked.connect(self.explain_with_Rise)
    
    def clean_lines(self):
        self.queryLine.clear()
        self.galleryLine.clear()

    def openQueryNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.queryLine.setText(fileName)
            self.pathQuery = fileName
            pixmap = QPixmap(self.pathQuery)
            self.query.setScaledContents(True)#para escalar la imagen al label
            self.query.setPixmap(pixmap)
            #print(fileName)

    def openGalleryNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getExistingDirectory(self, "Select Directory")
        #fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.galleryLine.setText(fileName)
            self.pathGallery = fileName
    
    def explain_with_Rise(self):
        #clear_layout = QtWidgets.QVBoxLayout()
        #self.widgetResult.setLayout(clear_layout)
        #print('layout: ', self.layout.count())
        #if (self.layout.count() == 1):
        #    self.layout.takeAt(self.layout.count()-1).
        
        self.waitLabel.setText('please, wait . . .')
        self.saliency_result = []
        self.progressBarMasks.setValue(0)
        self.progressBarExplain.setValue(0)

        query, x_query = xreid.load_img(60, 160 , self.pathQuery)
        #******** preprocesando os dados *********
        files = sorted(glob.glob( self.pathGallery + '/*.png'))
        #x_gallery = []
        #self.saliency_result = []

        x_q = np.repeat(x_query, batch_size, axis=0)

        for file in files[:20]:
            img, x = xreid.load_img(60, 160, file)
            x = np.repeat(x, batch_size, axis=0)
            
            predict = self.model_batch.run_on_batch(x_q, x)
            #predict_old = reid.predict_old(self.pathQuery, file)
            path_f, name_f = os.path.split(file)
            name = name_f.rsplit('.')
            #print(name)
            self.saliency_result.append([predict[0], img, x[0], None, name[0]])
            #self.saliency_result.append([predict_old[0], img, x[0], None, name[0]])
            print('predict ',predict[0])
        print(np.shape(self.saliency_result))
        # ****** resize batch on model *****
        #self.model_batch = None
        
        #tf.compat.v1.reset_default_graph()

        #self.model_batch = ReID_Model(batch_size)
        # ***** generate masks RISE ****** 
        sf, sc = 6, 4
        masks = xreid.generate_masks(self.model_batch, 2000, sf, sc, 0.5, self.progressBarMasks)
        #****** explicação do ReID  ***********
        #cv2.imwrite(masks,'mask.jpg')
        #plt.axis('off')
        #temp_img = x_query * masks[0]
        #plt.imshow( temp_img[0])
        #plt.imshow(sal[class_idx], cmap='jet', alpha=0.5)
        #plt.colorbar()
        #plt.show()

        x_q_batch = np.repeat(x_query, batch_size, axis=0)
        print(x_q_batch.shape)
        xreid.explain_reid(self.model_batch, batch_size, x_q_batch, self.saliency_result, masks, self.progressBarExplain)
        #****** sort saliencies maps
        print(np.shape(self.saliency_result))
        self.saliency_result.sort(key = lambda x: x[0] , reverse = True)
        self.saliency_result = np.array(self.saliency_result)

        print('sort: ', self.saliency_result[:, [0,4] ])
        #***** show results TOP 6 ****
        #f = plt.figure(figsize=(9,6))
        canvas_plt = MplCanvas(self, width=16, height=16, dpi=100)
        top = 20
        k = 0
        #***
        #while count < TIME_LIMIT:
        #    count += 1
        #    time.sleep(1)
        #    self.progress.setValue(count)
        for i in range(1, top+1):
            ax =  canvas_plt.fig.add_subplot(2,top, i)
            #plt.title('Explanation for `{}`'.format('ReID'))#class_name(class_idx)))
            ax.title.set_text( "{0:.0f}%".format(self.saliency_result[k][0] * 100) )
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(self.saliency_result[k][1])
            #im = ax.imshow(self.saliency_result[k][3], cmap='jet', alpha=0.5)
            #canvas_plt.fig.colorbar(im, ax=ax)
            #******************
            ax2 =  canvas_plt.fig.add_subplot(2,top, top + i)
            #plt.title('Explanation for `{}`'.format('ReID'))#class_name(class_idx)))
            ax2.title.set_text("{0:.0f}%".format(self.saliency_result[k][0] * 100))
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax2.imshow(self.saliency_result[k][1])
            im = ax2.imshow(self.saliency_result[k][3], cmap='jet', alpha=0.4)
            #canvas_plt.fig.colorbar(im, ax=ax2)
            
            k += 1
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(canvas_plt, self)

        #layout = QtWidgets.QVBoxLayout()
        #layout.addWidget(toolbar)
        #self.layout.deleteLater()
        if (self.layout.count() == 1):
            sz = self.layout.count() - 1
            self.layout.removeWidget(self.layout.takeAt(sz).widget())
        # removeItem() 
        self.layout.addWidget(canvas_plt)
        self.waitLabel.setText(' ')
        self.widgetResult.setLayout(self.layout)
    '''
        f.add_subplot(1,3, 1)
        plt.title('Query')#class_name(class_idx)))
        plt.imshow(img1)
        #plt.imshow(image1[0])
        f.add_subplot(1,3, 2)
        plt.title('Test Image')#class_name(class_idx)))
        plt.axis('off')
        plt.imshow(img2)
        f.add_subplot(1,3, 3)
        plt.title('Explanation for `{}`'.format('ReID'))#class_name(class_idx)))
        plt.axis('off')
        plt.imshow(img2)
        plt.imshow(sal[0], cmap='jet', alpha=0.5)
        plt.colorbar()
        #plt.imshow(image2[0])
        plt.show()
    '''
'''
    def calculate_tax(self):
        price = Decimal(self.price_box.text())
        tax = Decimal(self.tax_rate.value())
        total_price = price  + ((tax / 100) * price)
        total_price_string = "The total price with tax is: {:.2f}".format(total_price)
        self.results_output.setText(total_price_string)
'''
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())