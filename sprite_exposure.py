import numpy as np
import pandas as pd
import datetime as dt
import time
import random
from astropy.io import fits

from pylibftdi import Device

from matplotlib import pyplot as plt
import simulate_sprite_data as sprite_sim

# read in data and plot it in real time
class sprite_obs():

	def __init__(self, outname_df, outname_fits, detector_size=(2048,2048), 
					save_ttag=True, overwrite=True):

		self.outname_df = outname_df
		self.outname_fits = outname_fits
		self.save_ttag = save_ttag

		self.detector_size = detector_size
		self.exptime = 0.0

		# build dataframe to store incoming photons
		if overwrite: 
			self.ttag_df = pd.DataFrame(columns=['x', 'y', 'p', 'dt'])
			self.ttag_df.to_csv(self.outname_df, index=False)

		self.image_frame = np.zeros(self.detector_size)
		self.image_accum = np.zeros(self.detector_size)

		#build photon counters
		self.frame_count = 0
		self.accum_count = 0
		self.frame_count_lis = [0]
		self.accum_count_lis = [0] 
		self.ph_lis = np.array([])
		self.time_lis = [0]

		#build photon rate counters
		self.start_dt = dt.datetime.now()
		self.frame_rate = 0
		self.frame_rate_lis = []

	def ttag_to_image(self, data_df):
		im = np.zeros(self.detector_size)

		pix_df = data_df.groupby(['x', 'y']).count()
		pix_lis = pix_df.index.values
		pix_val_lis = pix_df['p'].values

		for i in range(len(pix_lis)):
			im[int(pix_lis[i][0])-1,int(pix_lis[i][1])-1] = pix_val_lis[i]

		return im

	def load_ttag(self, ttag_df):
		#build time list from dt column
		photon_ct_df = ttag_df.groupby(by='dt').count()
		ct_lis = photon_ct_df['x'].values
		time_lis = photon_ct_df.index.values

		accum_im = self.ttag_to_image(ttag_df)
		self.image_accum = im

		self.ph_lis = ttag_df['p'].values

	def update_image(self, dat_df):

		im = self.ttag_to_image(dat_df)
		self.image_frame = im 
		self.image_accum = self.image_accum + im

		#find photon count values
		self.frame_count = np.sum(self.image_frame)
		self.accum_count = np.sum(self.image_accum)

		self.frame_count_lis.append(self.frame_count)
		self.accum_count_lis.append(self.accum_count)

		end_dt = dt.datetime.now()
		self.frame_rate =  self.frame_count/(end_dt - self.start_dt).total_seconds()
		self.frame_rate_lis.append(self.frame_rate)
		self.start_dt = end_dt

		self.ph_lis = np.hstack([self.ph_lis, dat_df['p'].values])

		if self.save_ttag:
			with open(self.outname_df, 'a') as f:
				dat_df.to_csv(f, header=False, index=False)


	def aquire_data(self, read_name):

		#need to do whatever to read in the data type that should be saved to the input dat_path

		read_data = pd.read_csv(read_name).values

		dat_df = pd.DataFrame(read_data, columns=['x', 'y', 'p', 'dt'])

		self.update_image(dat_df)

		return dat_df


	def aquire_sim_data(self, sim_df_name, read_name='sim_dat.csv', photon_rate=1000):
		sim_obj = sprite_sim.sim_data(sim_df_name)

		sim_obj.simulate_data_flow(photon_rate=photon_rate, rand_photons=True,
									output_file=True, output_filename=read_name)

		dat_df = pd.read_csv(read_name)
		self.update_image(dat_df)

		return dat_df

	def save_accum(self, exp_tag='exp1'):
		outname = self.outname_fits.split('.fits')[0] + '_' + str(exp_tag) + '.fits'

		hdu = fits.PrimaryHDU(self.image_accum)
		hdu.writeto(outname)





