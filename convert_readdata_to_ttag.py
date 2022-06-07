import numpy as np
import pandas as pd
import datetime as dt

data_file_name = 'detector_in_capture_1_ftdi_read_bytes.txt'
data_out_name = 'test_detectorsim_data'

#open the file in a 'with' statement is it automatically closes when the script finishes executing
with open(data_file_name, 'r') as f:
	
	#break up the file by lines
	lines = f.readlines()
	print('Total Lines in File:', len(lines))

	xlis = []
	ylis = []
	plis = []
	tlis = []

	xbitlis = []
	ybitlis = []
	pbitlis = []
	llis = []

	extrabits = ''
	start_time = dt.datetime.now()

	# iterate through each line of the file which represents one readin time interval
	for i in range(len(lines)):

		# list containing the 8bit bytes in one line
		bytelis = lines[i][1:-2].split('\', \'')

		# join all the bytes together to make a single string of all bits
		# the [1:-1] part is to cut out the extra quotes it reads as a first and last character
		allbits_str = ''.join(bytelis)[1:-1]

		# if there were leftover bits from the last line append them to the front of the bit string
		allbits_str = extrabits + allbits_str

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

		# for this line make a list of timestamps for this line
		t_ints = [start_time+dt.timedelta(seconds=i)]*num_bitwords
		l_ints = [i]*num_bitwords

		# add the list of values for this line to the total lists
		xlis = xlis+x_ints
		ylis = ylis+y_ints
		plis = plis+p_ints
		tlis = tlis+t_ints

		xbitlis = xbitlis + x_bits
		ybitlis = ybitlis + y_bits
		pbitlis = pbitlis + p_bits
		llis = llis+l_ints

		# set the extrabits variable to be a string of the extra bits that were cut off
		extrabits = allbits_str[len(allbits_arr) - extra_bit_len::]

		# output the number of photons counted every 500 lines so we know it's doing something
		if (i+1) % 500 == 0:
			print('File Line:', i+1, '; Total Photons:', len(xlis))
		elif i == len(lines)-1:
			print('File Line:', i+1, '; Total Photons:', len(xlis))
			print(len(extrabits), 'Extra Bits Not Saved:', extrabits)

	# when all lines in the file are iterated through save the x, y, p, and dt lists to a csv file
	# this is saved in a format the SPRITE_GUI can read
	df = pd.DataFrame({'x':xlis, 'y':ylis, 'p':plis, 'dt':tlis})
	df.to_csv(data_out_name+'_TTAG.csv', index=False)
	print('TTAG File saved to:', data_out_name+'_TTAG.csv')


	# when all lines in the file are iterated through save the x, y, p, and dt lists to a csv file
	# This file can NOT be read by the SPRITE GUI
	df_bits = pd.DataFrame({'x':xbitlis, 'y':ybitlis, 'p':pbitlis, 'line':llis})
	df_bits.to_csv(data_out_name+'_BITS.csv', index=False)
	print('BIT File saved to:', data_out_name+'_BITS.csv')


