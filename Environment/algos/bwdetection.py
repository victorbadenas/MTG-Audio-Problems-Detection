import os
import sys
import numpy as np
import essentia.standard as estd
import scipy.signal
from scipy.interpolate import interp1d
from essentia import array as esarr
eps = np.finfo("float").eps

def detectBW(audio:list, SR:float, frame_size = 256, hop_size = 128, floor_db = -90, oversample_f = 1):
	
	frame_size *= oversample_f #if an oversample factor is desired, apply it

	fc_index_arr = []
	fft = estd.FFT(size = frame_size) #declare FFT function
	window = estd.Windowing(size=frame_size, type="hann") #declare windowing function

	for frame in estd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
		
		frame_fft = abs(fft(window(frame)))
		frame_fft_db = 20 * np.log10(frame_fft + eps) #calculate frame fft values in db
		
		interp_frame = compute_spectral_envelope(frame_fft_db, "linear") #compute the linear interpolation between the values of the maxima of the spectrum
		interp_frame = modify_floor(interp_frame, floor_db, log=True)

		fc_index = compute_fc(interp_frame)

		if energy_verification(frame_fft, fc_index): fc_index_arr.append(fc_index)

	if len(fc_index_arr) == 0: fc_index_arr = [frame_size]
	
	fc_bin, conf, binary = compute_mean_fc(fc_index_arr, np.arange(len(frame_fft)), SR)

	#print("mean_fc: ", fc_bin*SR/frame_size ," conf: ", conf ," binary_result: ", binary)

	return fc_bin*SR/frame_size, conf, binary

def is_power2(num:int):
	"""States if a number is a positive power of two

	Args:
		num: int to be ckecked

	Returns:
		(bool) True if is power of 2 false otherwise
	"""
	return num != 0 and ((num & (num - 1)) == 0)

def compute_spectral_envelope(frame_fft:list, kind="linear"):
	"""Compute the spectral envelope through a peak interpolation method

	Args:
		frame_fft: (list) iterable with the mono samples of a frame
	
	Kwargs:
		kind: (str) Specifies the kind of interpolation as a string 
			(‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, 
			‘previous’, ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and 
			‘cubic’ refer to a spline interpolation of zeroth, first, 
			second or third order; ‘previous’ and ‘next’ simply return the 
			previous or next value of the point) or as an integer specifying 
			the order of the spline interpolator to use. (Default = ‘linear’).

	Returns:
		(list) of the same length as frame_fft and xticks containing the 
			interpolated spectrum.
	"""
	xticks = np.arange(len(frame_fft))
	xp, hp = get_peaks(frame_fft, xticks) #compute the values of the local maxima of the function
	xp = np.append(xp,xticks[-1]) ; xp = np.append(0,xp) #appending the first value and last value in xticks
	hp = np.append(hp,frame_fft[-1]) ; hp = np.append(frame_fft[0],hp) #appending the first and last value of the function
	return interp1d(xp,hp,kind=kind)(xticks) #interpolating and returning

def get_peaks(sig:list, xticks:list):
	"""Returns the x,y values of the peaks in sig

	Args:
		sig: numpy.array of the signal of which to fing the peaks
		xticks: numpy.array of the corresponding x axis values for sig

	Returns:
		yval: y values of the peaks
		xval: x values of the peaks
	"""

	if len(sig) != len(xticks):
		raise ValueError("xticks and sig must have the same length")

	peaks, _ = scipy.signal.find_peaks(sig) #get the index values of the local maxima of sig

	tuplelist = [(a, b) for a, b in zip(xticks[peaks], sig[peaks])]
	tuplelist.sort(key=lambda x: x[1], reverse=True)

	xval = [a for a, b in tuplelist]
	yval = [b for a, b in tuplelist]

	return xval, yval

def modify_floor(sig:list, floor_db:float, log=False):
	"""Sets the values lower than a relative threshold to the threshold

	Args:
		sig: iterable with the values to compare
		floor_db: (float) containing the value in dB to be considered as a threshold
	
	Returns:
		sig: iterable containing the modified signal
	"""
	if log:
		sig[sig < (max(sig) + floor_db)] = max(sig) + floor_db
		return sig
	else:
		floor = 10 ** (floor_db/20)
		sig[sig < (max(sig) * floor)] = max(sig) * floor
		return sig

def compute_fc(interp_frame:list):
	"""Computes the biggest bin where the derivative goes below a threshold

	Args:
		interp_frame: iterable with frequency information

	Returns:
		(int) bin where the derivative crosses the th
	"""
	d = np.diff(interp_frame)[:-2]
	d = np.append(d,np.zeros(3))	
	return get_last_true_index(d <= (min(d) + 2))

def get_last_true_index(bool_list:list):
	"""Given a bool vector, returns the position of the last True value in the array

	Args:
		bool_list: iterable with boolean values

	Returns:
		(int) index position of the last True value in the array
	"""
	if not any(bool_list): return len(bool_list)-1
	
	for i,item in enumerate(reversed(bool_list)):
		if item: return len(bool_list)-1-i

def energy_verification(frame_fft:list, fc_index:int):
	"""Veryfies that the frame has at least 70% of the eneregy of the frame below fc_index

	Args:
		frame_fft: (iterable) with the frame's fft information
		fc_index: (iterable) limit for the energy calculation
	
	Returns:
		(boolean) True if the energy from 0 to fc_index is greater than the 70% of the energy of the frame
	"""
	return sum(frame_fft[:fc_index]**2)/sum(frame_fft**2) > 0.7

def compute_mean_fc(fc_index_arr:list, xticks:list, SR:float):
	"""computes the most possible fc for that audio file

	Args:
		fc_index_arr: iterable with the predicted fc's per frame
		xticks: iterable containing the bin2f array
		SR: (float) Sample Rate
	
	Returns:
		most_likely_f: (float) frequency corresponding to the highest peak in the histogram
		conf: (float) confidence value between 0 and 1
		(bool): True if the file is predicted to have the issue, False otherwise
	"""
	#fig,ax = plt.subplots(3,1,figsize=(15,9))
	hist = compute_histogram(fc_index_arr, xticks) #computation of the histogram
	most_likely_bin = np.argmax(hist) #bin value of the highest peak of the histogram

	#the confidence value changes depending on if most_likely_bin falls under the 90% lower spectrogram or not
	if most_likely_bin <= .9*len(hist):
		#if it falls under the 90% lower, the confidence is computed by a weighted sum of the values of the histogram,
		#the highest peak having the highest importance and decreasing as the indexes go further.

		#creation of the confidence scale
		#ax[0].stem(hist)
		conf_scale = abs(most_likely_bin - np.arange(len(hist))); conf_scale = max(conf_scale) - conf_scale ; conf_scale = conf_scale / max(conf_scale)
		#ax[1].stem(conf_scale)
		conf = sum(hist * conf_scale) / sum(hist) #computation of the confidence sum, normalised by the histogram length
		#ax[2].stem(hist * conf_scale / sum(hist))
		#plt.show()
		#return the analog frequency corresponding to the bin, confidence value, and True if the confidence value is higher than 0.6
		return most_likely_bin, conf, conf>0.6
	else:
		#if it falls over the 90% mark, the confidence is computated by summing the square of the 3 samples of the histogram closer to the max
		#and compare it to the sum of all the values appended to the histogram.
		#ax[0].stem(hist)
		if most_likely_bin+1 > len(hist)-1: conf = sum(hist[most_likely_bin-1:]**2)		
		else: conf = sum(hist[most_likely_bin-1:most_likely_bin+1]**2)
		conf /= sum(hist**2)
		#plt.show()
		return most_likely_bin, conf, False

def compute_histogram(idx_arr:list, xticks:list, mask = []):
	"""Computes the histogram of an array of ints

	Args:
		idx_arr: (iterable) with int values
		f: (iterable)x index array for the histogram
	
	KWargs:
		mask: (iterable) int binary array

	Returns:
		hist: histogram
	"""

	idx_arr = [int(item) for item in idx_arr]
	if len(mask) == 0:
		hist = np.zeros(len(xticks))
		for idx in idx_arr:
			hist[idx] += 1
		return hist
	else:
		hist = np.zeros(len(xticks))
		if len(mask) != len(idx_arr): raise ValueError("Inconsistent mask size")
		for idx, boolean in zip(idx_arr, mask):
			if boolean:
				hist[idx] += 1
		return hist
