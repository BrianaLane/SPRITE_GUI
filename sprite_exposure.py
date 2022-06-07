import numpy as np
import pandas as pd
import datetime as dt
import time
import random
from astropy.io import fits

#import ftd2xx as ftd

from matplotlib import pyplot as plt
import simulate_sprite_data as sprite_sim

# read in data and plot it in real time
class sprite_obs():

	def __init__(self, outname_df, outname_fits, detector_size=(2048,2048), 
					new_exp=True, save_ttag=True, overwrite=False):

		self.start_dt = dt.datetime.now()

		self.outname_df = outname_df
		self.outname_fits = outname_fits
		self.save_ttag = save_ttag

		self.detector_size = detector_size
		self.exptime = 0.0

		print(overwrite, self.outname_df)

		# build dataframe to store incoming photons
		if new_exp: 
			self.ttag_df = pd.DataFrame(columns=['x', 'y', 'p', 'dt'])
			
			if overwrite:
				self.outname_df = self.outname_df.split('.csv')[0]+'_TEST.csv'
				self.ttag_df.to_csv(self.outname_df, index=False)
			else:
				start_dt_str = self.start_dt.strftime("%d%m%Y_%H%M%S")
				self.outname_df = self.outname_df.split('.csv')[0]+'_'+start_dt_str+'.csv'
				self.ttag_df.to_csv(self.outname_df, index=False)

		self.image_frame = np.zeros(self.detector_size)
		self.image_accum = np.zeros(self.detector_size)

		#build photon counters
		self.frame_count = 0
		self.accum_count = 0
		self.frame_count_lis = [0]
		self.accum_count_lis = [0] 
		self.ph_lis = np.array([])
		self.elapsed_time = 0
		self.time_lis = [0]

		#build photon rate counters
		self.delay_delta = 0
		self.frame_rate = 0
		self.frame_phot_rate = 0
		self.frame_phot_rate_lis = []

		self.num_dat_updates = 0

		self.extra_read_bits = ''

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
		self.frame_count_lis = photon_ct_df['x'].values
		self.accum_count_lis = photon_ct_df['x'].expanding(1).sum().values

		self.frame_count = self.frame_count_lis[-1]
		self.accum_count = self.accum_count_lis[-1]

		self.datetime_lis = photon_ct_df.index.values
		timedelt_lis = [(i-self.datetime_lis[0])/np.timedelta64(1,'s') for i in self.datetime_lis]
		#this method of calculaing the time results in the first value being zero
		#need to update the values by the time of the first readout 
		#estimate time of first readout by taking median of difference between times and adding to time values
		frame_rate_lis = [timedelt_lis[i+1]-timedelt_lis[i] for i in range(len(timedelt_lis)-1)]
		self.frame_rate = np.median(frame_rate_lis)
		frame_rate_lis = [self.frame_rate]+frame_rate_lis
		self.time_lis = np.add(timedelt_lis, self.frame_rate)

		self.frame_phot_rate_lis = np.divide(self.frame_count_lis, frame_rate_lis)
		self.frame_phot_rate = self.frame_phot_rate_lis[-1]

		self.image_accum = self.ttag_to_image(ttag_df)
		#find only photons from last time frame
		frame_ttag_df = ttag_df.loc[self.datetime_lis[-1]]
		self.image_frame = self.ttag_to_image(frame_ttag_df)

		self.ph_lis = ttag_df['p'].values

	#Jack
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
		time_delta = (end_dt - self.start_dt).total_seconds()

		if self.num_dat_updates == 0:
			self.delay_delta = time_delta - self.frame_rate
			time_delta = time_delta - self.delay_delta

		self.elapsed_time = self.elapsed_time + time_delta
		self.time_lis.append(self.elapsed_time)

		self.frame_phot_rate =  self.frame_count/time_delta
		self.frame_phot_rate_lis.append(self.frame_phot_rate)
		self.start_dt = end_dt

		self.ph_lis = np.hstack([self.ph_lis, dat_df['p'].values])

		if self.save_ttag:
			with open(self.outname_df, 'a') as f:
				dat_df.to_csv(f, header=False, index=False)

		self.num_dat_updates += 1


	def bitstream_to_ttag(self, byte_list):

		bit_str = ''.join(byte_list)

		# if there were leftover bits from the last line append them to the front of the bit string
		allbits_str = self.extra_read_bits + bit_str

		# convert the string of all bits to an array of bits so the shape can be manipulated
		# each bit is still a string data type in the array
		allbits_arr = np.array(list(allbits_str))

		# take the total number of bits in the string mod 32
		# this is the extra number of bits at the end that don't make a complete 32 bit word 
		# these bits will get appended to the beginning of the next line's bit string
		extra_bit_len = len(allbits_arr)%32 
		
		# Remove the extra bits so the total number of bits is divisible by 32
		fullbits = allbits_arr[0: len(allbits_arr) - extra_bit_len]

		# reshape into 2D array so each line is 32 bits long
		bitwords = fullbits.reshape((-1, 32))
		# this is the number of complete 32 bit words in the line
		num_bitwords = np.shape(bitwords)[0]

		# join and save the first 12 bits of each line to X
		x_bits = [''.join(i) for i in bitwords[:, 0:12]]
		# join and save the second 12 bits of each line to Y
		y_bits = [''.join(i) for i in bitwords[:, 12:24]]
		# join and save the last 8 bits of each line to P
		p_bits = [''.join(i) for i in bitwords[:, 24::]]

		# convert each grouping of bits to an integer
		x_ints = [int(i,2) for i in x_bits]
		y_ints = [int(i,2) for i in y_bits]
		p_ints = [int(i,2) for i in p_bits]

		# set the extrabits variable to be a string of the extra bits that were cut off
		self.extra_read_bits = allbits_str[len(allbits_arr) - extra_bit_len::]

		return x_ints, y_ints, p_ints, self.extra_read_bits


	def aquire_data_fromFTDI(self, FTDI):

		byte_length = FTDI.getQueueStatus()
		byte_list = FTDI.read(byte_length)

		xlist, ylist, plist = self.bitstream_to_ttag(byte_list)

		timestamp = dt.datetime.now()
		dt_lis = [timestamp]*len(xlist)

		dat_df = pd.DataFrame({'x':xlist, 'y':ylist, 'p':plist, 'dt'_dt_lis})

		self.update_image(dat_df)

		return dat_df


	def aquire_sim_data(self, sim_df_name, photon_rate=1000):

		sim_obj = sprite_sim.sim_data(sim_df_name)

		dat_df = sim_obj.simulate_data_flow(photon_rate=photon_rate, sim_dt=False, 
											rand_photons=True, output_file=False)

		timestamp = dt.datetime.now()
		dt_lis = [timestamp]*len(dat_df)
		dat_df['dt'] = dt_lis

		self.update_image(dat_df)

		return dat_df

	def save_accum(self, exp_tag='exp1'):
		outname = self.outname_fits.split('.fits')[0] + '_' + str(exp_tag) + '.fits'

		hdu = fits.PrimaryHDU(self.image_accum)
		hdu.writeto(outname)



