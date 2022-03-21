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
        self.fileBtn = QtWidgets.QPushButton('Select Data')
        self.fileBtn.pressed.connect(self.select_data)

        self.currdatBtn = QtWidgets.QPushButton('Current Data')
        self.currdatBtn.pressed.connect(self.load_current_data)

        #build labels
        self.pv_data_label = QtWidgets.QLabel('Loaded Data: None')
        self.exptime_label_pv = QtWidgets.QLabel('Elapsed Time: '+str(self.elapsed_time)+' Seconds')
        self.frame_ct_label_pv = QtWidgets.QLabel('Photons per Second: '+str(self.exp_obj.frame_count) 
                                                + '\n' + 'Median Photons per Second: '
                                                + str(np.round(np.median(self.exp_obj.frame_count_lis), 2)))
        self.accum_ct_label_pv = QtWidgets.QLabel('Total Photons Accumulated: '+str(self.exp_obj.accum_count))

        #build figure canvas
        self.canvas_frame_pv = MplCanvas(self)
        self.plot_ref_frame_pv = None
        self.plot_ref_accum_pv = None

        #add button widgets
        self.tab2.grid.addWidget(self.fileBtn,0,0,1,2)
        self.tab2.grid.addWidget(self.fileBtn,1,0,1,2)

        #add label widgets
        self.tab2.grid.addWidget(self.pv_data_label,0,3,1,5)

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

    def select_data(self):
        filter = "CSV File (*.csv)"
        dlg = QtWidgets.QFileDialog.getOpenFileName(self, "CSV File", "./", filter)
        self.selectedData = pd.read_csv(dlg[0])

        self.pv_data_label.setText('Loaded Data: '+str(dlg[0]))

    def load_current_data(self):
        return None
        

    def read_save_ttag(self, s):
        if s == True:
            self.exp_obj.save_ttag = True
        if s == False:
            self.exp_obj.save_ttag = False

    def save_accum_image(self):
        curr_dt = datetime.datetime.now()
        curr_dt_str = curr_dt.strftime("%d%m%Y_%H%M%S")
        self.exp_obj.save_accum(exp_tag=curr_dt_str)

    def bin_image(self, image, bin_size):
        if bin_size == 1:
            return image

        else:
            num_bins = int(np.shape(image)[0]/bin_size)
            shape = (num_bins, image.shape[0] // num_bins,
                     num_bins, image.shape[1] // num_bins)
            return image.reshape(shape).sum(axis=(-1,1))


    def change_frame_bin(self, i):
        self.frame_bin = self.bin_lis[i]

    def change_accum_bin(self, i):
        self.accum_bin = self.bin_lis[i]

    def draw_image(self, ax, fig, im, bin_size, title):

        bin_im = self.bin_image(im, bin_size)

        plot_refs_f = ax.imshow(bin_im, origin='lower', cmap='gist_earth', vmin=0, vmax=10)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.03)
        cb = fig.colorbar(plot_refs_f, cax=cax, orientation='vertical')
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

    def draw_hist(self, ax, dat, title):

        ax.hist(dat, color='lightblue')
        ax.set_title(title, fontsize=6, pad=0.5, color='#ececec')
        ax.tick_params(axis='both', which='major', labelsize=3.5, color='#9fd3c7')
        ax.tick_params(axis='x', colors="#9fd3c7")
        ax.tick_params(axis='y', colors="#9fd3c7")

    def draw_plot(self, ax, xdata, ydata, title):

        plot_ref_p = self.canvas_frame.ax4.plot(xdata, ydata, color='lightblue', lw=2)
        ax.set_title(title, fontsize=6, pad=0.5, color='#ececec')
        ax.tick_params(axis='both', which='major', labelsize=3.5, color='#9fd3c7')
        ax.tick_params(axis='x', colors="#9fd3c7")
        ax.tick_params(axis='y', colors="#9fd3c7")
        return plot_ref_p[0]


    def update_figures(self):

        bin_image_frame = self.bin_image(self.exp_obj.image_frame, self.frame_bin)
        bin_image_accum = self.bin_image(self.exp_obj.image_accum, self.accum_bin)

        if np.max(self.exp_obj.image_accum) == 0:
            max_frame = 10
            max_accum = 10
        else:
            max_frame = np.max(bin_image_frame)
            max_accum = np.max(bin_image_accum)

        # We have a reference, we can use it to update the data for that line.
        self.plot_ref_frame.set_array(bin_image_frame)
        self.cbar_frame.mappable.set_clim(vmin=0, vmax=max_frame)
        self.bl_frame.set_text('Binned by: '+str(self.frame_bin))
        #cbar_frame_ticks = np.linspace(0., np.max(self.image_frame), num=5, endpoint=True)
        #self.cbar_frame.set_ticks(cbar_frame_ticks)
    
        self.plot_ref_accum.set_array(bin_image_accum)
        self.cbar_accum.mappable.set_clim(vmin=0, vmax=max_accum)
        self.bl_accum.set_text('Binned by: '+str(self.accum_bin))
        #cbar_accum_ticks = np.linspace(0., np.max(self.exp_obj.image_accum), num=5, endpoint=True)
        #self.cbar_accum.set_ticks(cbar_accum_ticks)

        self.canvas_frame.ax3.cla()
        self.draw_hist(self.canvas_frame.ax3, self.exp_obj.ph_lis, 'Pulse Height'+'\n'+'Histogram')

        self.plot_ref_ct.set_data(self.time_lis, self.exp_obj.accum_count_lis)
        self.canvas_frame.ax4.relim()
        self.canvas_frame.ax4.autoscale_view()

        # Trigger the canvas to update and redraw.
        self.cbar_frame.draw_all() 
        self.cbar_accum.draw_all() 
        self.canvas_frame.draw()

        # flush the GUI events
        self.canvas_frame.flush_events() 


    def build_figure_canvas(self, canvas):
        ref_frame, cbar_frame, bl_frame = self.draw_image(canvas.ax1, canvas.fig, 
                                                                self.exp_obj.image_frame, 1, 'Frame Rate Image')
        ref_accum, cbar_accum, bl_accum = self.draw_image(canvas.ax2, self.canvas_frame.fig, 
                                                                self.exp_obj.image_accum, 1, 'Accumulated Image')

        plot_ref_frame = (ref_frame, cbar_frame, bl_frame)
        plot_ref_accum= (ref_accum, cbar_accum, bl_accum)
        
        plot_ref_ct = self.draw_plot(canvas.ax4, [], [],'Accumulated'+'\n'+'Photons vs. Time')

        self.draw_hist(canvas.ax3, [], 'Pulse Height'+'\n'+'Histogram')



    
    def run_exposure(self):

        # Note: we no longer need to clear the axis.
        if self.plot_ref_frame is None:

            #build initial figures for first tab
            self.plot_ref_frame, self.cbar_frame, self.bl_frame = self.draw_image(self.canvas_frame.ax1, self.canvas_frame.fig, 
                                                                    self.exp_obj.image_frame, self.frame_bin, 'Frame Rate Image')
            self.plot_ref_accum, self.cbar_accum, self.bl_accum = self.draw_image(self.canvas_frame.ax2, self.canvas_frame.fig, 
                                                                    self.exp_obj.image_accum, self.accum_bin, 'Accumulated Image')
            
            self.plot_ref_ct = self.draw_plot(self.canvas_frame.ax4, self.time_lis, self.exp_obj.accum_count_lis,
                            'Accumulated'+'\n'+'Photons vs. Time')

            self.draw_hist(self.canvas_frame.ax3, self.exp_obj.ph_lis, 'Pulse Height'+'\n'+'Histogram')

            #build initial figures for second tab
            self.plot_ref_frame_pv, self.cbar_frame_pv, self.bl_frame_pv = self.draw_image(self.canvas_frame_pv.ax1, self.canvas_frame_pv.fig, 
                                                                    self.exp_obj.image_frame, self.frame_bin, 'Frame Rate Image')
            self.plot_ref_accum_pv, self.cbar_accum_pv, self.bl_accum_pv = self.draw_image(self.canvas_frame_pv.ax2, self.canvas_frame_pv.fig, 
                                                                    self.exp_obj.image_accum, self.accum_bin, 'Accumulated Image')
            
            self.plot_ref_ct_pv = self.draw_plot(self.canvas_frame_pv.ax4, [], [], 'Accumulated'+'\n'+'Photons vs. Time')

            self.draw_hist(self.canvas_frame_pv.ax3, [], 'Pulse Height'+'\n'+'Histogram')

        else:

            if self.runExposure:
                #count the elapsed time
                self.elapsed_time += self.readout_rate

                dat_df = self.exp_obj.aquire_sim_data(sim_df_name='example_simulated_gauss_ttag.csv', photon_rate=10000)

                self.time_lis.append(self.elapsed_time)

                #update labels
                self.exptime_label.setText('Elapsed Time: '+str(np.round(self.elapsed_time, 2))+' Seconds')
                self.frame_ct_label.setText('Photons per Second: '+str(int(self.exp_obj.frame_count)) 
                                            + '\n' + 'Median Photons per Second: '
                                            + str(np.round(np.median(self.exp_obj.frame_count_lis), 2)))
                self.accum_ct_label.setText('Total Photons Accumulated: '+str(int(self.exp_obj.accum_count)))

                self.update_figures()

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
        self.update_figures()

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

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(open("stylesheet.qss", "r").read())

    outname_df = 'ttag_exp_test.csv'
    outname_fits = 'ttag_exp_test.fits'

    w = MainWindow(outname_df=outname_df, outname_fits=outname_fits, readout_rate=1, detector_size=(2048,2048), overwrite=False)
    w.setWindowTitle("SPRITE GUI")
    w.setGeometry(0, 0, 1500, 800)

    app.exec_()
