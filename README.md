# SPRITE_GUI
A python dashboard for real time monitoring of the SPRITE cubesat detector readout

The SPRITE_GUI repository contains the following files:
	1. sprite_GUI.py: Contains a class for buiding the figure canvas in the GUI and the larger class the builds and runs the GUI 
	2. sprite_exposure.py: contains a class for building a sprite observation object (sprite_obs). These objects are created by the sprite_GUI.py MainWindow class and store and aquire data from either the SPRITE detector readout or from a simulated SPRITE observation. 
	3. simulate_sprite_data.py: contains a class for building a simulated sprite observation that is saved as csv file with ttag photon events. 
	4. stylesheet.qss: This file contains the stylesheet information for the pyQT package to read from to set the GUI style parameters for the various pyQT widgets used (ie. colors, fonts, etc..)
	5. example_simulated_gauss_ttag.csv: This is an example of an output set of simulated SPRITE data created with simulate_sprite_data.py. It contains 34K photons events from a random 2D guassian over the detector area. 

To run GUI in a terminal:
	cd your_path/SPRITE_GUI 
	python sprite_GUI.py

SPRITE GUI is currently set up to run with a set of simulated ttag data 
	(example_simulated_guass_ttag.csv)

SPRITE GUI Features:
	- Real_Time_Exposure Tab:
		1. 'Start/Stop Exposure' button (same button) run the simulated SPRITE observation. 
		2. The 'Reset Window' button will clear all of the GUI figures and exposure time. If exposure started again after reset the ttag data will continue to be saved to ttag file as long as 'Save TTAG' is checked. 
		3. The 'Save TTAG' check box will save all photon events to 'ttag_exp_test.csv'. New photon events will append to the end of the file and save after every accumulated set of data (now set to every 1 second). 
		4. The 'Save ACCUM' button will save a fits file of the displayed ACCUM Frame when pressed (with timestamp in filename). Note this is will not include the photons from before a reset. 
		5. 'Elapsed Time' shows the seconds passes since the exposure was started (resets to zero with window reset)
		6. 'Photons per Seconds' shows the rate of photons/second in the last accumulated frame of data (resets to zero with window reset)
		7. 'Median Photons per Seconds' shows the median rate of photons per second over the entire elasped time of the exposure (resets to zero with window reset)
		8. 'Total Photon Accumulated' show the total number of photons events during the entire elasped time of the exposure (resets to zero with window reset)
		9. The 'Frame Rate Image': shows the photons events mapped on the detector frame from the last accumulated frame. This frame clears by setting all pixels to zero after each accumulation. The colorbar on the right of the image shows the photon count per pixel. The scroll bar below changes the binning of the displayed image (new pixels per bin displayed at bottom left of figure ('Binned by: #')). The image will update with the new binning choosen when when the next accumulated frame is displayed. (frame is cleared with window reset)
		10. The 'ACCUM Rate Image': shows all of the photons events mapped on the detector frame from the entire exposure. This frame clears by setting all pixels to zero only when the window is reset. This image updates to include the new photons events after every accumulated frame. The colorbar on the right of the image shows the photon count per pixel. The scroll bar below changes the binning of the displayed image (new pixels per bin displayed at bottom left of figure ('Binned by: #')). The image will update with the new binning choosen when when the next accumulated frame is displayed. (frame is cleared with window reset)
		11. The 'Pulse Height Histogram' shows the distribution of pulse height values from all of the photons events during an exposure. (frame is cleared with window reset and updates after each accumulated frame)
		11. The 'Accumulated Photons vs. Time' shows the trend of accumulated photons over the elasped time of the exposure (frame is cleared with window reset and updates after each accumulated frame)

	- Real_Time_Exposure Tab:
		*** This tab currently doesn't do anything and is a work in progress ***
		1. All displayed widgets in this tab are in the Real_Time_Exposure Tab, however, this tab is for loading in the ttag csv file from a previous exposure. This tab provides a tool for displaying previous exposure data as a function of time. 
		2. In place of a Start/Stop/Reset button there is a 'Select Data' button that allows you to choose a '.csv' file containing ttag data from a previous exposure and load it into this tab

In Progress:
	- Real_Time_Exposure Tab:
		1. Add widget in GUI to allow the user to update the output file names of the ttag.csv file and the fits images. Right now it saves to same default name and the ttag.csv file gets overwritten with each run of the GUI. 
		2. Add widget in GUI to allow user to change data accumulation rate 
		3. Improve asthetics of the display
		4. Eventually build the functionality to aquire data from real time readout instead of a simulated data set. Placeholder function for this exists in sprite_exposure.py in the sprite_obs class with the acquire_data function. The sprite_GUI.py MainWindow class would require a one line update to read data from this function instead of the sprite_obs acquire_sim_data function is currently uses. 
	- Real_Time_Exposure Tab:
		1. Get the data to display on on widgets when loaded
		2. add scroll feature to allow user to scroll through the time of the exposure (need to figure out how to handle non-continouous ttag file) and efficiently assume data taking rate from dt column. 
