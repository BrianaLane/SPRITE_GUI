#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:41:43 2022

@author: brin8289

Ideas:
add buttons/dropdown on count rate plot to show frame ct vs time or accum ct vs time or both
add scale bar to change the binning of the images
add tab that allows plotting static images of old data (or snapshot frames over time window)

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

    def __init__(self, readout_rate, outname_df, outname_fits, detector_size=(2048,2048), overwrite=True):
        super(MainWindow, self).__init__()

        self.outname_df = outname_df
        self.outname_fits = outname_fits
        self.detector_size = detector_size
        self.overwrite = overwrite

        self.exp_obj = sprite_exp.sprite_obs(outname_df=self.outname_df, outname_fits=self.outname_fits,
                                             detector_size=self.detector_size, save_ttag=True, overwrite=self.overwrite)
        self.runExposure = False

        self.readout_rate = readout_rate
        self.detector_size = detector_size

        if overwrite:
            # build dataframe to store incoming photons
            ttag_df = pd.DataFrame(columns=['x', 'y', 'p', 'dt'])
            ttag_df.to_csv(self.outname_df, index=False)

        self.elapsed_time = 0.0
        self.time_lis = [0]

        self.frame_bin = 1
        self.accum_bin = 1
        self.bin_lis = [1, 8, 16, 32, 64, 128]

        self.grid = QtWidgets.QGridLayout(self)

        #build window tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        #self.tabs.resize(300,200)

        self.tabs.addTab(self.tab1,"Real Time Exposure")
        self.tabs.addTab(self.tab2,"Preview Exposure")
        self.tab1.grid = QtWidgets.QGridLayout(self)
        self.tab2.grid = QtWidgets.QGridLayout(self)

        #**************************#
        # first tab on main window #
        #**************************#

        #build labels
        self.exptime_label = QtWidgets.QLabel('Elapsed Time: '+str(self.elapsed_time)+' Seconds')
        self.frame_ct_label = QtWidgets.QLabel('Photons per Second: '+str(self.exp_obj.frame_count) 
                                                + '\n' + 'Median Photons per Second: '
                                                + str(np.round(np.median(self.exp_obj.frame_count_lis), 2)))
        self.accum_ct_label = QtWidgets.QLabel('Total Photons Accumulated: '+str(self.exp_obj.accum_count))

        #build save checkbox/button
        self.ttag_save_ch = QtWidgets.QRadioButton('Save TTAG')
        self.ttag_save_ch.setChecked(True)
        self.ttag_save_ch.toggled.connect(self.read_save_ttag)

        self.accum_save_Btn = QtWidgets.QPushButton('Save ACCUM')
        self.accum_save_Btn.pressed.connect(self.save_accum_image)

        #build control buttons
        self.startBtn = QtWidgets.QPushButton('Start Exposure')
        self.startBtn.pressed.connect(self.startExposure)

        self.restBtn = QtWidgets.QPushButton('Reset Window')
        self.restBtn.pressed.connect(self.resetWindow)

        #build image binning scale bars
        self.frame_slide = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slide.setMinimum(0)
        self.frame_slide.setMaximum(len(self.bin_lis)-1)
        self.frame_slide.setValue(0)
        self.frame_slide.setTickInterval(1)
        self.frame_slide.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.frame_slide.sliderMoved.connect(self.change_frame_bin)

        self.accum_slide = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.accum_slide.setMinimum(0)
        self.accum_slide.setMaximum(len(self.bin_lis)-1)
        self.accum_slide.setValue(0)
        self.accum_slide.setTickInterval(1)
        self.accum_slide.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.accum_slide.sliderMoved.connect(self.change_accum_bin)

        #Figure Grid
        self.canvas_frame = MplCanvas(self)
        self.plot_ref_frame = None
        self.plot_ref_accum = None

        #add button widgets
        self.tab1.grid.addWidget(self.startBtn,0,0,1,2)
        self.tab1.grid.addWidget(self.restBtn,1,0,1,2)

        #add save widgets
        self.tab1.grid.addWidget(self.ttag_save_ch,0,5,1,2)
        self.tab1.grid.addWidget(self.accum_save_Btn,1,5,1,2)

        #add figure widgets
        self.tab1.grid.addWidget(self.canvas_frame,2,0,10,14)

        #add scale bar bin image widgets
        self.tab1.grid.addWidget(self.frame_slide,10,1,1,2)
        self.tab1.grid.addWidget(self.accum_slide,10,6,1,2)

        #add label widgets
        self.tab1.grid.addWidget(self.exptime_label,0,8,1,2)
        self.tab1.grid.addWidget(self.frame_ct_label,13,0,1,2)
        self.tab1.grid.addWidget(self.accum_ct_label,13,5,1,2)


        #self.tab1.grid.setColumnStretch (column, stretch)
        #self.tab1.grid.setRowStretch (row, stretch)

        #***************************#
        # Second tab on main window #
        #***************************#

        #build control buttons
        self.loadexpBtn = QtWidgets.QPushButton('Load Current Exposure')
        self.fileBtn = QtWidgets.QPushButton('Load TTAG Data')
        self.loadexpBtn.pressed.connect(self.load_current_data)
        self.fileBtn.pressed.connect(self.load_ttag_data)

        #build labels
        self.pv_data_label = QtWidgets.QLabel('Loaded Data: None')
        self.exptime_label_pv = QtWidgets.QLabel('Elapsed Time: 0 Seconds')
        self.frame_ct_label_pv = QtWidgets.QLabel('Photons per Second: 0' + '\n' + 'Median Photons per Second: 0')
        self.accum_ct_label_pv = QtWidgets.QLabel('Total Photons Accumulated: 0')

        #build figure canvas
        self.canvas_frame_pv = MplCanvas(self)
        self.plot_ref_frame_pv = None
        self.plot_ref_accum_pv = None

        #add button widgets
        self.tab2.grid.addWidget(self.loadexpBtn,0,0,1,2)
        self.tab2.grid.addWidget(self.fileBtn,1,0,1,2)

        #add label widgets
        self.tab2.grid.addWidget(self.pv_data_label,0,3,1,5)
        self.tab2.grid.addWidget(self.exptime_label_pv,3,0,1,2)
        self.tab2.grid.addWidget(self.frame_ct_label_pv,13,0,1,2)
        self.tab2.grid.addWidget(self.accum_ct_label_pv,13,5,1,2)

        #add figure widgets
        self.tab2.grid.addWidget(self.canvas_frame_pv,2,0,10,14)

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

        # Add tabs to widget
        self.grid.addWidget(self.tabs)
        self.setLayout(self.grid)

        self.run_exposure()
        self.show()

    #TAB 1 functions

    def read_save_ttag(self, s):
        if s == True:
            self.exp_obj.save_ttag = True
        if s == False:
            self.exp_obj.save_ttag = False

    def save_accum_image(self):
        curr_dt = datetime.datetime.now()
        curr_dt_str = curr_dt.strftime("%d%m%Y_%H%M%S")
        self.exp_obj.save_accum(exp_tag=curr_dt_str)

    def change_frame_bin(self, i):
        self.frame_bin = self.bin_lis[i]

    def change_accum_bin(self, i):
        self.accum_bin = self.bin_lis[i]

    # def update_figures(self, canvas):

    #     bin_image_frame = canvas.bin_image(self.exp_obj.image_frame, self.frame_bin)
    #     bin_image_accum = canvas.bin_image(self.exp_obj.image_accum, self.accum_bin)

    #     if np.max(self.exp_obj.image_accum) == 0:
    #         max_frame = 10
    #         max_accum = 10
    #     else:
    #         max_frame = np.max(bin_image_frame)
    #         max_accum = np.max(bin_image_accum)

    #     canvas.frame_im_ref.set_array(bin_image_frame)
    #     canvas.frame_cb_ref.mappable.set_clim(vmin=0, vmax=max_frame)
    #     canvas.frame_bl_ref.set_text('Binned by: '+str(self.frame_bin))

    #     canvas.accum_im_ref.set_array(bin_image_accum)
    #     canvas.accum_cb_ref.mappable.set_clim(vmin=0, vmax=max_accum)
    #     canvas.accum_bl_ref.set_text('Binned by: '+str(self.accum_bin))

    #     canvas.ax3.cla()
    #     canvas.draw_hist(self.exp_obj.ph_lis, 'Pulse Height'+'\n'+'Histogram')

    #     canvas.plot_ref.set_data(self.time_lis, self.exp_obj.accum_count_lis)
    #     canvas.ax4.relim()
    #     canvas.ax4.autoscale_view()

    #     # Trigger the canvas to update and redraw.
    #     canvas.frame_cb_ref.draw_all() 
    #     canvas.accum_cb_ref.draw_all() 
    #     canvas.draw()

    #     # flush the GUI events
    #     canvas.flush_events() 
    
    def run_exposure(self):

        # Note: we no longer need to clear the axis.
        if self.canvas_frame.frame_im_ref is None:

            print('DRAW FIGURES')

            self.canvas_frame.draw_frame_image(self.exp_obj.image_frame, self.frame_bin)
            self.canvas_frame.draw_accum_image(self.exp_obj.image_accum, self.accum_bin)
            self.canvas_frame.draw_plot(self.time_lis, self.exp_obj.accum_count_lis, 'Accumulated'+'\n'+'Photons vs. Time')
            self.canvas_frame.draw_hist(self.exp_obj.ph_lis, 'Pulse Height'+'\n'+'Histogram')

            #build initial figures for second tab
            self.canvas_frame_pv.draw_frame_image(self.exp_obj.image_frame, self.frame_bin)
            self.canvas_frame_pv.draw_accum_image(self.exp_obj.image_accum, self.accum_bin)
            self.canvas_frame_pv.draw_plot([], [], 'Accumulated'+'\n'+'Photons vs. Time')
            self.canvas_frame_pv.draw_hist([], 'Pulse Height'+'\n'+'Histogram')

        else:

            if self.runExposure:
                #count the elapsed time
                self.elapsed_time += self.readout_rate

                dat_df = self.exp_obj.aquire_sim_data(sim_df_name='example_simulated_gauss_ttag.csv', photon_rate=10000)

                self.time_lis.append(self.elapsed_time)

                #update labels
                self.exptime_label.setText('Elapsed Time: '+str(np.round(self.elapsed_time, 2))+' Seconds')
                self.frame_ct_label.setText('Photons per Second: '+str(int(self.exp_obj.frame_rate)) 
                                            + '\n' + 'Median Photons per Second: '
                                            + str(np.round(np.median(self.exp_obj.frame_rate_lis), 2)))
                self.accum_ct_label.setText('Total Photons Accumulated: '+str(int(self.exp_obj.accum_count)))

                self.canvas_frame.update_figures(self.exp_obj.image_frame, self.exp_obj.image_accum, 
                                                 self.frame_bin, self.accum_bin, self.exp_obj.ph_lis, 
                                                 self.exp_obj.time_lis, self.exp_obj.accum_count_lis)

    def startExposure(self):
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
        self.time_lis = [0]

        self.frame_bin = 1
        self.accum_bin = 1
        self.frame_slide.setValue(0)
        self.accum_slide.setValue(0)

        #update figures
        self.canvas_frame.update_figures(self.exp_obj.image_frame, self.exp_obj.image_accum, 
                                         self.frame_bin, self.accum_bin, self.exp_obj.ph_lis, 
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

    #TAB 2 Functions

    def dateparse_df(self, dt):    
        return pd.Timestamp(dt)

    def update_ttag_preview(self, pv_df):

        self.pv_obj.load_ttag(pv_df)

        self.pv_data_label.setText('Loaded Data: '+str(self.pv_obj.outname_df))
        self.exptime_label_pv.setText('Elapsed Time: '+str(np.round(self.pv_obj.time_lis[-1], 1))+' Seconds')
        self.frame_ct_label_pv.setText('Photons per Second: '+str(np.round(self.pv_obj.frame_rate,1)) 
                                                + '\n' + 'Median Photons per Second: '
                                                + str(np.round(np.median(self.pv_obj.frame_rate_lis), 2)))
        self.accum_ct_label_pv.setText('Total Photons Accumulated: '+str(int(self.pv_obj.accum_count)))

    def load_current_data(self):
        self.pv_data = pd.read_csv(self.outname_df, parse_dates=True, date_parser=self.dateparse_df, index_col='dt')

        self.pv_obj = sprite_exp.sprite_obs(outname_df=self.outname_df, outname_fits=self.outname_fits,
                                             detector_size=self.detector_size, save_ttag=False, overwrite=False)
        self.update_ttag_preview(self.pv_data)

    def load_ttag_data(self):
        dlg = QtWidgets.QFileDialog.getOpenFileName(self, "CSV File", "./", "CSV files (*.csv)")

        self.pv_data = pd.read_csv(dlg[0], parse_dates=True, date_parser=self.dateparse_df, index_col='dt')

        self.pv_obj = sprite_exp.sprite_obs(outname_df=dlg[0], outname_fits=self.outname_fits,
                                             detector_size=self.detector_size, save_ttag=False, overwrite=False)
        self.update_ttag_preview(self.pv_data)

#Build and run GUI when script is run
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(open("stylesheet.qss", "r").read())

    outname_df = 'ttag_exp_test.csv'
    outname_fits = 'ttag_exp_test.fits'

    w = MainWindow(outname_df=outname_df, outname_fits=outname_fits, readout_rate=1, detector_size=(2048,2048), overwrite=False)
    w.setWindowTitle("SPRITE GUI")
    w.setGeometry(0, 0, 1500, 800)

    app.exec_()
