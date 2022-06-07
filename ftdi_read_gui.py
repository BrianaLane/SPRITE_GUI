import sys
import numpy as np
import random
import time
import ftd2xx as ftd

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPalette, QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

class MainWindow(QtWidgets.QWidget):

   def __init__(self):
      super(MainWindow, self).__init__()

      
      self.setFixedWidth(700)
      self.setFixedHeight(700)

      self.filename = 'ftdi_read_bytes.txt'

      self.d = ftd.open(0)
      print('open device')

      info = self.d.getDeviceInfo()
      print('Device:', info['description'])

      self.baudrate = 115200
      self.num_bits = 8
      self.stop_bits = 1 
      self.parity = 0 #None
      self.flowcontrol = 0

      self.d.setBaudRate(self.baudrate)
      self.d.setDataCharacteristics(self.num_bits, self.stop_bits, self.parity)
      self.d.setFlowControl(self.flowcontrol)

      self.grid = QtWidgets.QGridLayout(self)

      # counters
      self.num_bytes = 0
      self.dat = []
      self.tot_bytes = 0
      # creating flag
      self.flag = False
      self.contread = True
      self.fileopen = False
      self.dat_type = 'int'

      self.totlabel = QtWidgets.QLabel('Total Bytes Recorded: '+str(self.tot_bytes))
      self.totlabel.setFont(QFont('Arial', 25))

      self.datlabel = QtWidgets.QLabel('Data Saved to '+self.filename)
      self.datlabel.resize(120, 80)
      self.datlabel.adjustSize()

      self.hexdata_ch = QtWidgets.QCheckBox('Hex Data')
      self.hexdata_ch.setChecked(False)

      self.intdata_ch = QtWidgets.QCheckBox('Integer Data')
      self.intdata_ch.setChecked(False)

      self.bindata_ch = QtWidgets.QCheckBox('Binary Data')
      self.bindata_ch.setChecked(True)

      self.hexdata_ch.stateChanged.connect(self.clickhex)
      self.intdata_ch.stateChanged.connect(self.clickint)
      self.bindata_ch.stateChanged.connect(self.clickbin)

      self.read1Btn = QtWidgets.QPushButton("Read")
      self.read1Btn.clicked.connect(self.read_once)

      self.startBtn = QtWidgets.QPushButton("Start Continuous Read")
      self.startBtn.clicked.connect(self.start_read)

      self.resetBtn = QtWidgets.QPushButton("Clear Data")
      self.resetBtn.clicked.connect(self.clear_data)

      self.closeBtn = QtWidgets.QPushButton("Close FTDI")
      self.closeBtn.clicked.connect(self.close_ftdi)

      self.openBtn = QtWidgets.QPushButton("Open FTDI")
      self.openBtn.clicked.connect(self.open_ftdi)

      self.purgeBtn = QtWidgets.QPushButton("Purge FTDI Buffers")
      self.purgeBtn.clicked.connect(self.purge_ftdi)

      self.statBtn = QtWidgets.QPushButton("Queue Status")
      self.statBtn.clicked.connect(self.queue_ftdi)

      #baud rate selection box
      self.br_box = QtWidgets.QComboBox()
      self.baudrate_options = [9600, 14400, 19200, 38400, 57600, 115200, 128000, 256000]
      self.br_box.addItems([str(i) for i in self.baudrate_options])
      self.br_box.activated[str].connect(self.select_BaudRate)

      self.br_label = QtWidgets.QLabel('Baud Rate: '+str(self.baudrate))

      #add button widgets
      self.grid.addWidget(self.totlabel,0,0,1,2)
      self.grid.addWidget(self.datlabel,1,0,1,2)

      self.grid.addWidget(self.read1Btn,5,0,1,2)
      self.grid.addWidget(self.startBtn,6,0,1,2)
      self.grid.addWidget(self.resetBtn,7,0,1,2)

      self.grid.addWidget(self.statBtn,8,0,1,1)
      self.grid.addWidget(self.purgeBtn,8,1,1,1)

      self.grid.addWidget(self.openBtn, 9,0,1,1)
      self.grid.addWidget(self.closeBtn,9,1,1,1)

      self.grid.addWidget(self.br_box, 10,0,1,1)
      self.grid.addWidget(self.br_label, 10,1,1,1)

      self.grid.addWidget(self.hexdata_ch,11,0,1,1)
      self.grid.addWidget(self.intdata_ch,12,0,1,1)
      self.grid.addWidget(self.bindata_ch,13,0,1,1)

      self.timer=QtCore.QTimer()
      self.timer.timeout.connect(self.show_data_256)
      self.timer.start(0)

      self.setLayout(self.grid)
      self.show()

   def show_data(self):
      # checking if flag is true
      if self.flag:
         byte_length = self.d.getQueueStatus()
         self.dat = self.d.read(byte_length)
         self.num_bytes = len(self.dat)
         self.tot_bytes = self.tot_bytes + self.num_bytes

         if self.num_bytes > 0:
            dat_int = [bytearray(self.dat)[i] for i in range(self.num_bytes)]
            if self.dat_type == 'int':
               self.dat = dat_int
            elif self.dat_type == 'bin':
               self.dat = [f'{i:08b}' for i in dat_int]
     
         if self.contread:
            if byte_length > 0:
               self.wf.write(str(self.dat)+'\n')
               self.totlabel.setText('Total Bytes Recorded: '+str(self.tot_bytes))

         else:
            self.datlabel.setText('Data:(' + str(len(self.dat))+ ')' + str(self.dat))
            self.totlabel.setText('Total Bytes Recorded: '+str(self.num_bytes))
            self.flag = False

      def show_data_256(self):
      # checking if flag is true
      if self.flag:
         self.dat = self.d.read(256)
         self.num_bytes = len(self.dat)
         self.tot_bytes = self.tot_bytes + self.num_bytes

         dat_int = [bytearray(self.dat)[i] for i in range(self.num_bytes)]
         if self.dat_type == 'int':
            self.dat = dat_int
         elif self.dat_type == 'bin':
            self.dat = [f'{i:08b}' for i in dat_int]
  
         if self.contread:
            self.wf.write(str(self.dat)+'\n')
            self.totlabel.setText('Total Bytes Recorded: '+str(self.tot_bytes))

         else:
            self.datlabel.setText('Data:(' + str(len(self.dat))+ ')' + str(self.dat))
            self.totlabel.setText('Total Bytes Recorded: '+str(self.num_bytes))
            self.flag = False

   def start_read(self):
      self.contread = True

      if self.startBtn.text() == "Pause Continuous Read":
         self.startBtn.setText("Resume Continuous Read")
         self.flag = False

      else:
         if self.startBtn.text() == "Start Continuous Read":
            self.datlabel.setText('Data Saved to '+self.filename)

            if not self.fileopen:
               self.wf = open(self.filename, 'w')
               self.fileopen = True

         self.flag = True
         self.startBtn.setText("Pause Continuous Read")

   def read_once(self):
      self.flag = True
      self.contread = False

   def clear_data(self):
      self.flag = False

      self.num_bytes = 0
      self.dat = []
      self.tot_bytes = 0
      self.totlabel.setText('Total Bytes Recorded: '+str(self.tot_bytes))
      self.datlabel.setText('Data:' )
      self.startBtn.setText("Start Continuous Read")

      if self.fileopen:
         self.wf.close()
      self.wf = open(self.filename, 'w')
      self.wf.close()

      self.fileopen = False
      self.contread = False
      print("read output stopped; file closed")  

   def queue_ftdi(self):
      buff_bytes = self.d.getQueueStatus()
      self.datlabel.setText('Queue Status: '+ str(buff_bytes) + ' bytes')

   def purge_ftdi(self):
      self.d.purge()
      time.sleep(1)
      self.d.purge()

      b_bytes = self.d.getQueueStatus()
      self.datlabel.setText('Purging Buffers 5 Times: '+ str(b_bytes) + ' bytes left')
      print('FTDI Buffers Purged') 

   def select_BaudRate(self, i):
      self.baudrate = int(i)
      self.d.setBaudRate(self.baudrate)
      time.sleep(1)
      self.br_label.setText('Baud Rate: '+str(self.baudrate))
      print('Baud Rate Changed to ', str(self.baudrate))

   def clickhex(self, state):
      if state == QtCore.Qt.Checked:
         self.dat_type = 'hex'
         self.intdata_ch.setChecked(False)
         self.bindata_ch.setChecked(False)

   def clickint(self, state):
      if state == QtCore.Qt.Checked:
         self.dat_type = 'int'
         self.hexdata_ch.setChecked(False)
         self.bindata_ch.setChecked(False)

   def clickbin(self, state):
      if state == QtCore.Qt.Checked:
         self.dat_type = 'bin'
         self.hexdata_ch.setChecked(False)
         self.intdata_ch.setChecked(False)

   def open_ftdi(self):
      self.d = ftd.open(0)
      print('FTDI is Open')

      info = self.d.getDeviceInfo()
      print('Device:', info['description'])

      self.d.setBaudRate(self.baudrate)
      self.d.setDataCharacteristics(self.num_bits, self.stop_bits, self.parity)
      self.d.setFlowControl(self.flowcontrol)

   def close_ftdi(self):
      self.d.close()
      print('FTDI is Closed')

#Build and run GUI when script is run
if __name__ == "__main__":
   app = QtWidgets.QApplication(sys.argv)

   w = MainWindow()
   w.setWindowTitle("Readout")
   w.setGeometry(50,50,320,200)

   app.exec_()
