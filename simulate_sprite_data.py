#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:17:53 2022

@author: Briana Indahl
"""

import numpy as np
import pandas as pd
import datetime as dt
import time
import random
from astropy.io import fits

from matplotlib import pyplot as plt
import seaborn as sns

# builds a data frame of simulated photon events
# simulate_data_flow allows user to obtain random groupings of
#   data points over a # seconds delay
class sim_data():

    def __init__(self, sim_df_name='sim_ttag_test.csv', detector_size=(2048,2048)):
        
        self.detector_size = detector_size

        self.sim_df_name = sim_df_name
        self.sim_photon_df = pd.read_csv(sim_df_name)
        
        if not set(['x', 'y', 'p']).issubset(set(self.sim_photon_df.columns)):
            raise ValueError('DataFrame must have columns [x, y, p]')
            
        self.n_photons = len(self.sim_photon_df)
        
    
    def rand_pulse_heights(self, p_mu=20, p_sig=2, noise=False):
        p_vals = np.random.normal(p_mu, p_sig, self.n_photons)
        return p_vals
    
    
    def build_sim_random(self, npts=1e5, p_mu=20, p_sig=2, pulse_noise=False,
                                   save=True, savename='sim_ttag_test.csv'):
        self.n_photons = npts
        rand_x_vals = np.random.randint(low=0, high=self.detector_size[0], size=npts)
        rand_y_vals = np.random.randint(low=0, high=self.detector_size[1], size=npts)
        
        p_vals = self.rand_pulse_heights(p_mu=p_mu, p_sig=p_sig, noise=pulse_noise)
        
        self.sim_photon_df = pd.DataFrame(columns=['x', 'y', 'p'])
        
        self.sim_photon_df['x'] = rand_x_vals
        self.sim_photon_df['y'] = rand_y_vals
        self.sim_photon_df['p'] = p_vals
        
        if save: 
            print('SAVING simulated data to:', savename)
            self.sim_photon_df.to_csv(savename, index=False)
        
   
    #Builds 2D random guassian ball of photons on detector center
    def build_sim_guass(self, npts=1e5, std=1000, p_mu=20, p_sig=2, pulse_noise=False,
                                  save=True, savename='sim_ttag_test.csv'):
        
        guass_mean = (int(self.detector_size[0]/2), int(self.detector_size[0]/2))
        guess_std = std * np.eye(2)
        guass_photons = np.random.multivariate_normal(guass_mean, guess_std,
                                                      int(npts))
        
        #bin photons into pixels simply by rounding
        rand_photons = np.round(guass_photons)
    
        #find all photons that fall within detector size
        valid_x = (rand_photons[:, 0] <= self.detector_size[0]) & (rand_photons[:, 0] > 0)
        valid_y = (rand_photons[:, 1] <= self.detector_size[1]) & (rand_photons[:, 1] > 0)
        valid_pts = valid_x & valid_y
        
        valid_photons = rand_photons[valid_pts, :]
        self.n_photons = np.shape(valid_photons)[0]
        
        p_vals = self.rand_pulse_heights(p_mu=p_mu, p_sig=p_sig, noise=pulse_noise)
        
        self.sim_photon_df = pd.DataFrame(columns=['x', 'y', 'p'])
        
        self.sim_photon_df['x'] = valid_photons[:,0]
        self.sim_photon_df['y'] = valid_photons[:,1]
        self.sim_photon_df['p'] = p_vals
        
        if save: 
            print('SAVING simulated data to:', savename)
            self.sim_photon_df.to_csv(savename, index=False)
            
    
    def build_sim_fromimage(self, fits_name, p_mu=20, p_sig=2,
                            pulse_noise=False, save=True,
                            savename='sim_ttag_test.csv'):
        
        hdu = fits.open(fits_name)
        dat = hdu[0].data
        hdu.close()

        self.detector_size = np.shape(dat)

        self.n_photons = np.sum(dat)
        self.sim_photon_df = pd.DataFrame(columns=['x', 'y', 'p'])
        
        photon_coords = []
        for x in np.arange(np.shape(dat)[1]):
            for y in np.arange(np.shape(dat)[0]):
                pix_photons = dat[y,x]
                pix_coords = [(y, x), ]*pix_photons
                photon_coords = photon_coords+pix_coords
                
        photon_coords_shuffle = random.sample(photon_coords,
                                              len(photon_coords))  

        x_vals = [i[1] for i in photon_coords_shuffle]
        y_vals = [i[0] for i in photon_coords_shuffle]
        
        p_vals = self.rand_pulse_heights(p_mu=p_mu, p_sig=p_sig,
                                         noise=pulse_noise)
                
        self.sim_photon_df['x'] = x_vals
        self.sim_photon_df['y'] = y_vals
        self.sim_photon_df['p'] = p_vals
        
        if save: 
            print('SAVING simulated data to:', savename)
            self.sim_photon_df.to_csv(savename, index=False)
            
            
    def plot_sim_photons(self, save=False, savepath='sim_sprite_data.png'):
        if len(self.sim_photon_df) > 0:
            
            fig, axes = plt.subplots(1,2, figsize=(10,4))

            sns.histplot(data=self.sim_photon_df, x="x", y="y",
                         discrete=True, binrange=(0,self.detector_size[0]), cbar=True, ax=axes[0])
            axes[1].hist(self.sim_photon_df['p'], bins=100)

            axes[0].set_title('Photons per Pixel', fontsize=20)
            axes[1].set_title('Pulse Height Distribution', fontsize=20)
            
            if save: 
                plt.savefig(savepath)
            
            plt.show()
                
        else:
            print('No simulated photon data built')
            
    
    def simulate_data_flow(self, photon_rate=10, delay=0, start_ind=0,
                           rand_photons=False, output_file=False, output_filename='/sim_dat.csv',
                           verbose=False):

        start = time.time()
        start_time = dt.datetime.now()
        
        if rand_photons:
            rand_inds = np.random.randint(low=start_ind, high=self.n_photons, size=photon_rate)
            dat_df = self.sim_photon_df.iloc[rand_inds].copy()
            
        else:
            dat_df = self.sim_photon_df.iloc[start_ind: start_ind+photon_rate].copy()
        
        dat_df['dt'] = start_time
        
        time.sleep(delay)
        end = time.time()

        if verbose:
            print(end-start, 'sec')

        if output_file:
            dat_df.to_csv(output_filename, index=False)
            return None
        
        return dat_df.values



