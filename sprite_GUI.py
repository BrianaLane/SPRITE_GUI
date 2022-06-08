#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:41:43 2022

@author: brin8289

Ideas:
add buttons/dropdown on count rate plot to show frame ct vs time or accum ct vs time or both
"""

import sys
import random
import matplotlib
import numpy as np
import pandas as pd
import datetime as datetime
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPalette

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize)

#import ftd2xx as ftd

import simulate_sprite_data as sprite_sim
import sprite_exposure as sprite_exp



class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=8, height=4.5, dpi=200):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        gs = self.fig.add_gridspec(2, 3, width_ratios=[6,6,4], height_ratios=[1.5,1], wspace=0.4, hspace=0.5)
        self.ax1 = self.fig.add_subplot(gs[:,0])
        self.ax2 = self.fig.add_subplot(gs[:,1])
        self.ax3 = self.fig.add_subplot(gs[0,2])
        self.ax4 = self.fig.add_subplot(gs[1,2])
        self.fig.patch.set_facecolor('#323232')
        super(MplCanvas, self).__init__(self.fig)

        self.plot_ref = None

        self.frame_im_ref = None
        self.frame_cb_ref = None
        self.frame_bl_ref = None

        self.accum_im_ref = None
        self.accum_cb_ref = None
        self.accum_bl_ref = None

    def bin_image(self, image, bin_size):
        if bin_size == 1:
            return image

        else:
            num_bins = int(np.shape(image)[0] / bin_size)
            shape = (num_bins, image.shape[0] // num_bins,
                     num_bins, image.shape[1] // num_bins)
            return image.reshape(shape).sum(axis=(-1,1))

    def draw_image(self, ax, im, bin_size, title):

        bin_im = self.bin_image(im, bin_size)

        plot_refs_f = ax.imshow(bin_im, origin='lower', cmap='gist_earth', vmin=0, vmax=10)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.03)
        cb = self.fig.colorbar(plot_refs_f, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=4, color='white')
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_color("white")

        ax.set_title(title, fontsize=9, color='#ececec')
        bin_label = ax.text(-0.2, -0.5, 'Binned by: '+str(bin_size), fontsize=6, color='#ececec')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        for s in ax.spines.values():
            s.set_edgecolor('#4f4f4f')

        return plot_refs_f, cb, bin_label

    def draw_frame_image(self, im, bin_size):
        frame_ref_lis = self.draw_image(self.ax1, im, bin_size, 'Frame Rate Image')
        self.frame_im_ref, self.frame_cb_ref, self.frame_bl_ref = frame_ref_lis

    def draw_accum_image(self, im, bin_size):
        accum_ref_lis = self.draw_image(self.ax2, im, bin_size, 'Accumulated Image')
        self.accum_im_ref, self.accum_cb_ref, self.accum_bl_ref = accum_ref_lis

    def draw_hist(self, dat, title):
        self.ax3.hist(dat, color='lightblue')
        self.ax3.set_title(title, fontsize=6, pad=0.5, color='#ececec')
        self.ax3.tick_params(axis='both', which='major', labelsize=3.5, color='#9fd3c7')
        self.ax3.tick_params(axis='x', colors="#9fd3c7")
        self.ax3.tick_params(axis='y', colors="#9fd3c7")

    def draw_plot(self, xdata, ydata, title):
        plot_ref_all = self.ax4.plot(xdata, ydata, color='lightblue', lw=2)
        self.plot_ref = plot_ref_all[0]
        self.ax4 .set_title(title, fontsize=6, pad=0.5, color='#ececec')
        self.ax4 .tick_params(axis='both', which='major', labelsize=3.5, color='#9fd3c7')
        self.ax4 .tick_params(axis='x', colors="#9fd3c7")
        self.ax4 .tick_params(axis='y', colors="#9fd3c7")

    def update_figures(self, frame_im, accum_im, frame_bin, accum_bin, ph_lis, time_lis, accum_ct_lis):

        bin_image_frame = self.bin_image(frame_im, frame_bin)
        bin_image_accum = self.bin_image(accum_im, accum_bin)

        if np.max(frame_im) == 0:
            max_frame = 10
            max_accum = 10
        else:
            max_frame = np.max(bin_image_frame)
            max_accum = np.max(bin_image_accum)

        self.frame_im_ref.set_array(bin_image_frame)
        self.frame_cb_ref.mappable.set_clim(vmin=0, vmax=max_frame)
        self.frame_bl_ref.set_text('Binned by: '+str(frame_bin))

        self.accum_im_ref.set_array(bin_image_accum)
        self.accum_cb_ref.mappable.set_clim(vmin=0, vmax=max_accum)
        self.accum_bl_ref.set_text('Binned by: '+str(accum_bin))

        self.ax3.cla()
        self.draw_hist(ph_lis, 'Pulse Height'+'\n'+'Histogram')

        self.plot_ref.set_data(time_lis, accum_ct_lis)
        self.ax4.relim()
        self.ax4.autoscale_view()

        # Trigger the canvas to update and redraw.
        self.frame_cb_ref.draw_all() 
        self.accum_cb_ref.draw_all() 
        self.draw()

        # flush the GUI events
        self.flush_events() 

class MainWindow(QtWidgets.QWidget):

    def __init__(self, readout_rate, outname_df, outname_fits, detector_size=(2048,2048)):
        super(MainWindow, self).__init__()

        self.outname_df = outname_df
        self.outname_base_df = outname_df
        self.outname_fits = outname_fits

        self.readout_rate = readout_rate
        self.detector_size = detector_size

        self.initialize_figure = False
        self.initialize_exposure = False
        self.runExposure = False
        self.overwrite = False

        self.elapsed_time = 0.0
        self.elap_time_lis = [0]

        self.frame_bin_t1 = 32
        self.accum_bin_t1 = 32
        self.frame_bin_t2 = 32
        self.accum_bin_t2 = 32
        self.bin_lis = [1, 8, 16, 32, 64, 128]

        self.pv_dat_filename = 'PV'

        # set default FTDI parameters
        self.baudrate = 115200
        self.num_bits = 8
        self.stop_bits = 1 
        self.parity = 0 #None
        self.flowcontrol = 0

        self.FTDI_open = False
        self.FTDI_info = ''
        self.open_ftdi()

        if self.FTDI_open:
            self.DATA_MODE = 'ftdi'
        else:
            self.DATA_MODE = 'sim'

        self.mode_params = {'sim': {'title':'SIMULATION MODE', 
                                    'color': '#e0773f',
                                    't3_mess': 'No FTDI Connection Found'},
                            'ftdi':{'title': 'CONNECTED TO FTDI: '+self.FTDI_info, 
                                    'color': '#50C66F',
                                    't3_mess': 'CONNECTED TO FTDI: '+self.FTDI_info}}

        self.grid = QtWidgets.QGridLayout(self)

        #build window tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()

        self.tabs.addTab(self.tab1,"Real Time Exposure")
        self.tabs.addTab(self.tab2,"Preview Exposure")
        self.tabs.addTab(self.tab3,"FTDI Parameters")
        self.tab1.grid = QtWidgets.QGridLayout(self)
        self.tab2.grid = QtWidgets.QGridLayout(self)
        self.tab3.grid = QtWidgets.QGridLayout(self)

        #**************************#
        # first tab on main window #
        #**************************#

        self.mode_label = QtWidgets.QLabel(self.mode_params[self.DATA_MODE]['title'])
        title_col = self.mode_params[self.DATA_MODE]['color']
        self.mode_label.setAlignment(QtCore.Qt.AlignCenter)
        self.mode_label.setStyleSheet('font-size: 40; font-weight: bold; border: 5px solid black; background-color: '+title_col) 

        self.exptime_label = QtWidgets.QLabel('Elapsed Time: '+str(self.elapsed_time)+' Seconds')
        self.frame_ct_label = QtWidgets.QLabel('Photons per Second: '+str(0.0) 
                                                + '\n' + 'Median Photons per Second: ' + str(0.0))
        self.accum_ct_label = QtWidgets.QLabel('Total Photons Accumulated: '+str(0))

        #build save checkbox/button
        self.ttag_save_ch = QtWidgets.QCheckBox('Save TTAG')
        self.ttag_save_ch.setChecked(True)
        self.ttag_save_ch.toggled.connect(self.read_save_ttag)

        #build overwrite checkbox/button
        self.ttag_overwrite_ch = QtWidgets.QCheckBox('Overwrite Test')
        self.ttag_overwrite_ch.setChecked(False)
        self.ttag_overwrite_ch.toggled.connect(self.set_overwrite)

        self.accum_save_Btn = QtWidgets.QPushButton('Save ACCUM')
        self.accum_save_Btn.pressed.connect(self.save_accum_image)

        #build control buttons
        self.startBtn = QtWidgets.QPushButton('Start Exposure')
        self.startBtn.pressed.connect(self.startExposure)

        self.restBtn = QtWidgets.QPushButton('Reset Exposure')
        self.restBtn.pressed.connect(self.resetWindow)

        #build image binning scale bars
        self.frame_slide_t1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slide_t1.setMinimum(0)
        self.frame_slide_t1.setMaximum(len(self.bin_lis)-1)
        self.frame_slide_t1.setValue(3)
        self.frame_slide_t1.setTickInterval(1)
        self.frame_slide_t1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.frame_slide_t1.sliderMoved.connect(self.change_frame_bin_t1)
        self.frame_slide_t1.setGeometry(10,10,8,20)

        self.accum_slide_t1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.accum_slide_t1.setMinimum(0)
        self.accum_slide_t1.setMaximum(len(self.bin_lis)-1)
        self.accum_slide_t1.setValue(3)
        self.accum_slide_t1.setTickInterval(1)
        self.accum_slide_t1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.accum_slide_t1.sliderMoved.connect(self.change_accum_bin_t1)
        self.frame_slide_t1.setGeometry(10,10,8,20)

        #Figure Grid
        self.canvas_frame = MplCanvas(self)
        self.plot_ref_frame = None
        self.plot_ref_accum = None

        #add button widgets
        self.tab1.grid.addWidget(self.startBtn,0,0,1,2)
        self.tab1.grid.addWidget(self.restBtn,1,0,1,2)

        #add save widgets
        self.tab1.grid.addWidget(self.ttag_save_ch,0,3,1,1)
        self.tab1.grid.addWidget(self.ttag_overwrite_ch,0,4,1,1)
        self.tab1.grid.addWidget(self.accum_save_Btn,1,3,1,2)

        #add figure widgets
        self.tab1.grid.addWidget(self.canvas_frame,2,0,3,9)

        #add scale bar bin image widgets
        self.tab1.grid.addWidget(self.frame_slide_t1,5,1,1,2)
        self.tab1.grid.addWidget(self.accum_slide_t1,5,4,1,2)

        #add label widgets
        self.tab1.grid.addWidget(self.mode_label,0,6,1,3)
        self.tab1.grid.addWidget(self.exptime_label,1,6,1,3)
        self.tab1.grid.addWidget(self.frame_ct_label,7,0,1,3)
        self.tab1.grid.addWidget(self.accum_ct_label,7,3,1,3)

        self.tab1.grid.setRowStretch(0,1)
        self.tab1.grid.setRowStretch(1,1)
        self.tab1.grid.setRowStretch(2,50)
        self.tab1.grid.setRowStretch(3,50)
        self.tab1.grid.setRowStretch(4,50)
        self.tab1.grid.setRowStretch(5,1)
        self.tab1.grid.setRowStretch(6,1)
        self.tab1.grid.setRowStretch(7,1)

        for c in range(9):
            self.tab1.grid.setColumnStretch(c,1)


        #***************************#
        # Second tab on main window #
        #***************************#

        #build control buttons
        self.loadexpBtn = QtWidgets.QPushButton('Load Current Exposure')
        self.fileBtn = QtWidgets.QPushButton('Load TTAG Data')
        self.saveBtn_t2 = QtWidgets.QPushButton('Save ACCUM')
        self.loadexpBtn.pressed.connect(self.load_current_data)
        self.fileBtn.pressed.connect(self.load_ttag_data)
        self.saveBtn_t2.pressed.connect(self.save_accum_image_t2)

        #build labels
        self.pv_data_label = QtWidgets.QLabel('Loaded Data: None')
        self.exptime_label_pv = QtWidgets.QLabel('Elapsed Time: 0 Seconds')
        self.frame_ct_label_pv = QtWidgets.QLabel('Photons per Second: 0' + '\n' + 'Median Photons per Second: 0')
        self.accum_ct_label_pv = QtWidgets.QLabel('Total Photons Accumulated: 0')

        #build figure canvas
        self.canvas_frame_pv = MplCanvas(self)
        self.plot_ref_frame_pv = None
        self.plot_ref_accum_pv = None

        #build image binning scale bars
        self.frame_slide_t2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slide_t2.setMinimum(0)
        self.frame_slide_t2.setMaximum(len(self.bin_lis)-1)
        self.frame_slide_t2.setValue(3)
        self.frame_slide_t2.setTickInterval(1)
        self.frame_slide_t2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.frame_slide_t2.sliderMoved.connect(self.change_frame_bin_t2)
        self.frame_slide_t2.setGeometry(10,10,8,20)

        self.accum_slide_t2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.accum_slide_t2.setMinimum(0)
        self.accum_slide_t2.setMaximum(len(self.bin_lis)-1)
        self.accum_slide_t2.setValue(3)
        self.accum_slide_t2.setTickInterval(1)
        self.accum_slide_t2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.accum_slide_t2.sliderMoved.connect(self.change_accum_bin_t2)
        self.accum_slide_t2.setGeometry(10,10,8,20)

        #build exposure time scale bar 
        self.exptime_slide = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exptime_slide.setTickInterval(1)
        self.exptime_slide.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.exptime_slide.sliderMoved.connect(self.change_pv_exptime)

        #add button widgets
        self.tab2.grid.addWidget(self.loadexpBtn,0,0,1,3)
        self.tab2.grid.addWidget(self.fileBtn,1,0,1,3)
        self.tab2.grid.addWidget(self.saveBtn_t2,0,5,1,2)

        #add label widgets
        self.tab2.grid.addWidget(self.exptime_label_pv,0,3,1,2)
        self.tab2.grid.addWidget(self.pv_data_label,1,3,1,3)

        self.tab2.grid.addWidget(self.frame_ct_label_pv,7,0,2,3)
        self.tab2.grid.addWidget(self.accum_ct_label_pv,7,3,2,3)

        #add figure widgets
        self.tab2.grid.addWidget(self.canvas_frame_pv,2,0,3,9)

        #add scale bar bin image widgets
        self.tab2.grid.addWidget(self.frame_slide_t2,5,1,1,2)
        self.tab2.grid.addWidget(self.accum_slide_t2,5,4,1,2)

        self.tab2.grid.setRowStretch(0,1)
        self.tab2.grid.setRowStretch(1,1)
        self.tab2.grid.setRowStretch(2,50)
        self.tab2.grid.setRowStretch(3,50)
        self.tab2.grid.setRowStretch(4,50)
        self.tab2.grid.setRowStretch(5,1)
        self.tab2.grid.setRowStretch(6,1)
        self.tab2.grid.setRowStretch(7,1)

        for c in range(9):
            self.tab2.grid.setColumnStretch(c,1)

        #**************************#
        # Third tab on main window #
        #**************************#

        self.baudrate = 115200
        self.num_bits = 8
        self.stop_bits = 1 
        self.parity = 0 #None
        self.flowcontrol = 0

        self.ftdi_status = QtWidgets.QLabel(self.mode_params[self.DATA_MODE]['t3_mess'])
        #self.ftdi_connect_stat.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.ftdi_status.setAlignment(QtCore.Qt.AlignCenter)
        self.ftdi_status.setStyleSheet('font-size: 40; font-weight: bold; border: 5px solid black; background-color: '+title_col) 

        self.ftdicloseBtn = QtWidgets.QPushButton("Close FTDI")
        self.ftdicloseBtn.clicked.connect(self.close_ftdi)

        self.ftdiopenBtn = QtWidgets.QPushButton("Open FTDI")
        self.ftdiopenBtn.clicked.connect(self.open_ftdi)

        self.ftdistatBtn = QtWidgets.QPushButton("Queue Status")
        self.ftdistatBtn.clicked.connect(self.queue_ftdi)

        self.ftdipurgeBtn = QtWidgets.QPushButton("Purge FTDI Buffers")
        self.ftdipurgeBtn.clicked.connect(self.purge_ftdi)

        #baud rate selection box
        self.ftdi_br_box = QtWidgets.QComboBox()
        self.baudrate_options = [9600, 14400, 19200, 38400, 57600, 115200, 128000, 256000]
        self.ftdi_br_box.addItems([str(i) for i in self.baudrate_options])
        self.ftdi_br_box.activated[str].connect(self.select_BaudRate)

        self.ftdi_br_label = QtWidgets.QLabel('Baud Rate: '+str(self.baudrate))

        #add button widgets
        self.tab3.grid.addWidget(self.ftdi_status,0,1,1,6)

        self.tab3.grid.addWidget(self.ftdiopenBtn,1,0,1,2)
        self.tab3.grid.addWidget(self.ftdicloseBtn,1,3,1,2)

        self.tab3.grid.addWidget(self.ftdistatBtn,2,3,1,2)

        self.tab3.grid.addWidget(self.ftdipurgeBtn,3,0,1,2)

        self.tab3.grid.addWidget(self.ftdi_br_box,4,0,1,2)
        self.tab3.grid.addWidget(self.ftdi_br_label,4,3,1,2)

        for c in range(7):
            self.tab3.grid.setColumnStretch(c,1)

        for r in range(5):
            self.tab3.grid.setRowStretch(r,1)

        #**********************#
        # build GUI and Timers #
        #**********************#

        # # Setup a timer to trigger the redraw by calling update_plot.
        # Create timer object
        self.timer = QtCore.QTimer(self)
        # Add a method with the timer
        self.timer.timeout.connect(self.run_exposure)
        # Call start() method to modify the timer value
        self.timer.start(self.readout_rate*1000)

        self.tab1.setLayout(self.tab1.grid)
        self.tab2.setLayout(self.tab2.grid)
        self.tab3.setLayout(self.tab3.grid)

        # Add tabs to widget
        self.grid.addWidget(self.tabs)
        self.setLayout(self.grid)

        self.run_exposure()
        self.show()

    #TAB 1 functions

    def set_overwrite(self):
        self.overwrite = self.ttag_overwrite_ch.isChecked()

    def read_save_ttag(self):
        if self.initialize_exposure:
            self.exp_obj.save_ttag = self.ttag_save_ch.isChecked()


    def save_accum_image(self):
        if self.initialize_exposure:
            curr_dt = datetime.datetime.now()
            curr_dt_str = curr_dt.strftime("%d%m%Y_%H%M%S")
            self.exp_obj.save_accum(exp_tag=curr_dt_str)


    def change_frame_bin_t1(self, i):
        self.frame_bin_t1 = self.bin_lis[i]


    def change_accum_bin_t1(self, i):
        self.accum_bin_t1 = self.bin_lis[i]

    
    def run_exposure(self):

        # Note: we no longer need to clear the axis.
        #if self.canvas_frame.frame_im_ref is None:
        if not self.initialize_figure:

            print('DRAW FIGURES')

            self.canvas_frame.draw_frame_image(np.zeros(self.detector_size), self.frame_bin_t1)
            self.canvas_frame.draw_accum_image(np.zeros(self.detector_size), self.accum_bin_t1)
            self.canvas_frame.draw_plot([], [], 'Accumulated'+'\n'+'Photons vs. Time')
            self.canvas_frame.draw_hist([], 'Pulse Height'+'\n'+'Histogram')

            #build initial figures for second tab
            self.canvas_frame_pv.draw_frame_image(np.zeros(self.detector_size), self.frame_bin_t2)
            self.canvas_frame_pv.draw_accum_image(np.zeros(self.detector_size), self.accum_bin_t2)
            self.canvas_frame_pv.draw_plot([], [], 'Accumulated'+'\n'+'Photons vs. Time')
            self.canvas_frame_pv.draw_hist([], 'Pulse Height'+'\n'+'Histogram')

            self.initialize_figure=True

        else:

            if self.runExposure:
                #count the elapsed time
                self.elapsed_time += self.readout_rate

                if self.DATA_MODE == 'sim':
                    dat_df = self.exp_obj.aquire_sim_data(sim_df_name='example_simulated_gauss_ttag.csv', photon_rate=10000)
                elif self.DATA_MODE == 'ftdi':
                    dat_df = self.exp_obj.aquire_ftdi_data(FTDI=self.FTDI)

                self.elap_time_lis.append(self.elapsed_time)

                #update labels
                self.exptime_label.setText('Elapsed Time: '+str(np.round(self.exp_obj.time_lis[-1], 2))+' Seconds')
                self.frame_ct_label.setText('Photons per Second: '+str(int(self.exp_obj.frame_phot_rate)) 
                                            + '\n' + 'Median Photons per Second: '
                                            + str(np.round(np.median(self.exp_obj.frame_phot_rate_lis), 2)))
                self.accum_ct_label.setText('Total Photons Accumulated: '+str(int(self.exp_obj.accum_count)))

                self.canvas_frame.update_figures(self.exp_obj.image_frame, self.exp_obj.image_accum, 
                                                 self.frame_bin_t1, self.accum_bin_t1, self.exp_obj.ph_lis, 
                                                 self.exp_obj.time_lis, self.exp_obj.accum_count_lis)


    def startExposure(self):

        if not self.initialize_exposure:
            # build SPRITE exposure object 
            self.exp_obj = sprite_exp.sprite_obs(outname_df=self.outname_base_df, outname_fits=self.outname_fits,
                                             detector_size=self.detector_size, new_exp=True, 
                                             save_ttag=self.ttag_save_ch.isChecked(), 
                                             overwrite=self.overwrite)
            self.outname_df = self.exp_obj.outname_df
            self.exp_obj.frame_rate = self.readout_rate
            self.initialize_exposure = True

            self.ttag_overwrite_ch.setCheckable(False)
            self.ttag_overwrite_ch.setStyleSheet("QCheckBox::indicator:hover {border: 2px solid #8a8a8a;}")

        # Set the caption of the start button based on previous caption
        if self.startBtn.text() == 'Stop Exposure':
            self.startBtn.setText('Resume Exposure')
            self.runExposure = False
        else:
            # making startWatch to true 
            self.runExposure = True
            self.startBtn.setText('Stop Exposure')


    def resetWindow(self):
        self.runExposure = False

        # Reset all counter time variables and set image to zeros
        self.elapsed_time = 0.0
        self.exp_obj.image_frame = np.zeros(self.detector_size)
        self.exp_obj.image_accum = np.zeros(self.detector_size)

        self.exp_obj.frame_count = 0
        self.exp_obj.accum_count = 0
        self.exp_obj.frame_count_lis = [0]
        self.exp_obj.accum_count_lis = [0]
        self.exp_obj.ph_lis = np.array([])
        self.exp_obj.elapsed_time = 0
        self.exp_obj.time_lis = [0]
        self.exp_obj.num_dat_updates = 0

        self.elap_time_lis = [0]

        self.frame_bin_t1 = 32
        self.accum_bin_t1 = 32
        self.frame_slide_t1.setValue(3)
        self.accum_slide_t1.setValue(3)

        #update figures
        self.canvas_frame.update_figures(self.exp_obj.image_frame, self.exp_obj.image_accum, 
                                         self.frame_bin_t1, self.accum_bin_t1, self.exp_obj.ph_lis, 
                                         self.exp_obj.time_lis, self.exp_obj.accum_count_lis)

        # Set the initial values for the stop watch
        self.exptime_label.setText('Elapsed Time: '+str(self.elapsed_time)+' Seconds')
        self.frame_ct_label.setText('Photons per Second: '+str(int(self.exp_obj.frame_count)) 
                                    + '\n' + 'Median Photons per Second: '
                                    + str(np.round(np.median(self.exp_obj.frame_count_lis), 2)))
        self.accum_ct_label.setText('Total Photons Accumulated: '+str(int(self.exp_obj.accum_count)))
        
        #reset button label
        self.startBtn.setText('Start Exposure')

        # Trigger the canvas to update and redraw.
        self.canvas_frame.draw()

        # flush the GUI events
        self.canvas_frame.flush_events() 

        self.initialize_exposure = False
        self.ttag_overwrite_ch.setCheckable(True)
        self.ttag_overwrite_ch.setChecked(False)
        self.overwrite = False 
        self.ttag_overwrite_ch.setStyleSheet("color: white")
        self.ttag_overwrite_ch.setStyleSheet("QCheckBox::indicator:hover {border: 2px solid #9fd3c7;}")

    #TAB 2 Functions

    def dateparse_df(self, dt):    
        return pd.Timestamp(dt)


    def update_ttag_preview(self):

        time_lis = self.pv_obj.time_lis[0:self.time_ind]
        phot_per_sec = self.pv_obj.frame_phot_rate_lis[self.time_ind]
        med_phot_per_sec = np.median(self.pv_obj.frame_phot_rate_lis[0:self.time_ind])
        accum_ct_lis = self.pv_obj.accum_count_lis[0:self.time_ind]

        self.pv_data_label.setStyleSheet('background-color: #323232')
        self.pv_data_label.setText('Loaded Data: '+str(self.pv_dat_filename).split('/')[-1])
        self.exptime_label_pv.setText('Elapsed Time: '+str(np.round(time_lis[-1], 1))+' Seconds')
        self.frame_ct_label_pv.setText('Photons per Second: '+str(np.round(phot_per_sec,1)) 
                                                + '\n' + 'Median Photons per Second: '
                                                + str(np.round(med_phot_per_sec, 2)))
        self.accum_ct_label_pv.setText('Total Photons Accumulated: '+str(int(accum_ct_lis[-1])))

        pv_datetime = self.pv_obj.datetime_lis[self.time_ind]

        pv_data_reidx = self.pv_data.copy().reset_index()
        dt_ind_vals = pv_data_reidx[pv_data_reidx['dt']==pv_datetime].index.values

        #find only photons from last time frame
        pv_frame_df = self.pv_data.iloc[dt_ind_vals[0]:dt_ind_vals[-1]+1]
        image_frame = self.pv_obj.ttag_to_image(pv_frame_df)

        pv_accum_df = self.pv_data.iloc[0:dt_ind_vals[-1]+1]
        image_accum = self.pv_obj.ttag_to_image(pv_accum_df)

        ph_lis = pv_accum_df['p'].values

        self.canvas_frame_pv.update_figures(image_frame, image_accum, self.frame_bin_t2, self.accum_bin_t2,
                                            ph_lis, time_lis, accum_ct_lis)


    def load_current_data(self):
        if self.initialize_exposure:
            dat_filename = self.exp_obj.outname_df
            self.initialize_data_preview(dat_filename)


    def load_ttag_data(self):
        dlg = QtWidgets.QFileDialog.getOpenFileName(self, "CSV File", "./", "CSV files (*.csv)")
        dat_filename = dlg[0]
        self.initialize_data_preview(dat_filename)


    def initialize_data_preview(self, dat_filename):
        
        try:
            self.pv_data = pd.read_csv(dat_filename, parse_dates=True, date_parser=self.dateparse_df, index_col='dt')
            self.pv_dat_filename = dat_filename

            if len(self.pv_data) == 0: 
                self.pv_data_label.setStyleSheet('background-color: #CD4A3D')
                self.pv_data_label.setText('Loaded Data: EMPTY ARRAY')
                return('No Preview Data Loaded')

        except FileNotFoundError:
            self.pv_data_label.setStyleSheet('background-color: #CD4A3D')
            self.pv_data_label.setText('Loaded Data: FILE NOT FOUND')
            return('No Preview Data Loaded')

        self.pv_obj = sprite_exp.sprite_obs(outname_df=dat_filename, outname_fits=self.outname_fits,
                                             detector_size=self.detector_size, new_exp=False, 
                                             save_ttag=False, overwrite=False)

        self.pv_obj.load_ttag(self.pv_data)

        slide_range = len(self.pv_obj.time_lis)-1
        self.time_ind = slide_range

        self.exptime_slide.setMinimum(1)
        self.exptime_slide.setMaximum(slide_range)
        self.exptime_slide.setValue(slide_range)
        self.exptime_slide.setStyleSheet("QSlider::groove:horizontal {width: 1000px}")
        self.tab2.grid.addWidget(self.exptime_slide,6,0,1,8)
        
        self.update_ttag_preview()


    def save_accum_image_t2(self):
        frame_dt = self.pv_obj.datetime_lis[self.time_ind]
        dts = str(frame_dt)
        frame_dt_str = dts[9:10]+dts[5:7]+dts[0:4]+'_'+dts[11:13]+dts[14:16]+dts[17::].split('.')[0]
        filename = self.pv_dat_filename.split('/')[-1].split('.csv')[0]
        self.pv_obj.save_accum(exp_tag=filename+'_'+frame_dt_str)


    def change_frame_bin_t2(self, i):
        self.frame_bin_t2 = self.bin_lis[i]
        self.update_ttag_preview()


    def change_accum_bin_t2(self, i):
        self.accum_bin_t2 = self.bin_lis[i]
        self.update_ttag_preview()


    def change_pv_exptime(self, i):
        self.time_ind = i
        self.update_ttag_preview()
        

    # TAB 3 Functions

    def open_ftdi(self):
        try:
            self.FTDI = ftd.open(0)
            self.FTDI_open = True
        except NameError:
            self.FTDI_open = False
            print('NO FTDI Device Found')
            print('GUI in SIMULATE MODE')

        if self.FTDI_open:
            self.FTDIinfo = self.d.getDeviceInfo()

            self.FTDI.setBaudRate(self.baudrate)
            self.FTDI.setDataCharacteristics(self.num_bits, self.stop_bits, self.parity)
            self.FTDI.setFlowControl(self.flowcontrol)

    def close_ftdi(self):
        if self.FTDI_open:
            self.FTDI.close()
        self.FTDI_open = False

    def queue_ftdi(self):
        if self.FTDI_open:
            buff_bytes = self.FTDI.getQueueStatus()
            self.datlabel.setText('Queue Status: '+ str(buff_bytes) + ' bytes')

    def purge_ftdi(self):
        if self.FTDI_open:
            self.FTDI.purge()
            time.sleep(1)
            self.FTDI.purge()

            b_bytes = self.FTDI.getQueueStatus()
            self.datlabel.setText('Purging Buffers 5 Times: '+ str(b_bytes) + ' bytes left')

    def select_BaudRate(self, i):
        self.baudrate = int(i)
        if self.FTDI_open:
            self.FTDI.setBaudRate(self.baudrate)
            time.sleep(1)
        self.ftdi_br_label.setText('Baud Rate: '+str(self.baudrate))


#Build and run GUI when script is run
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(open("stylesheet.qss", "r").read())

    outname_df = 'ttag_exp.csv'
    outname_fits = 'ttag_exp.fits'

    w = MainWindow(outname_df=outname_df, outname_fits=outname_fits, readout_rate=1, detector_size=(4096,4096))
    w.setWindowTitle("SPRITE GUI")
    w.setGeometry(0, 0, 1500, 800)

    app.exec_()
