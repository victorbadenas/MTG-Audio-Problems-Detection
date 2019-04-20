import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import essentia.standard as estd
import scipy.signal
from essentia import array as esarr
eps = np.finfo("float").eps

from scipy.interpolate import interp1d

def is_power2(num:int):
	"""States if a number is a positive power of two

	Args:
		num: int to be ckecked

	Returns:
		(bool) True if is power of 2 false otherwise
	"""
	return num != 0 and ((num & (num - 1)) == 0)

def normalise(sig:list, log=False):
	"""Normalises the values in a list

	Args:
		sig: iterable containing the values

	KWargs:
		log: (boolean) True if the list contains values in dB, False otherwise

	Returns:
		normalised list
	"""
	return np.array(sig) / max(sig) if not log else (np.array(sig) - max(sig))

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

def compute_fc(interp_frame:list):
	
	d = np.diff(interp_frame)[:-2]
	d = np.append(d,np.zeros(3))
	#d[d>0] = 0
	#d[d!=min(d)]=0

	#plt.plot(d)
	#plt.show()
	
	return get_last_true_index( d <= (min(d) + 2) )

def compute_mean_fc(hist:list, fc_index_arr:list, f:list, SR:float):
	most_likely_bin = np.argmax(hist)
	mean_fc = f[most_likely_bin]
	#print("len(f):",len(f))
	#print("most_likely_bin:",most_likely_bin)
	std_bin = int( min(np.std(fc_index_arr), len(f)/12) )
	#std = std_bin/len(f)*(SR/2)
	#print("std_bin:",std_bin)
	#conf = hist * abs(np.arange(len(hist)) - most_likely_bin) / ( max(abs(np.arange(len(hist)) - most_likely_bin)) * sum(hist))
	hist = hist/sum(hist)
	if most_likely_bin-int(std_bin/2) < 0:
		conf = sum( hist[:most_likely_bin+int(std_bin/2)] )
	elif most_likely_bin+int(std_bin/2) > len(hist):
		conf = sum( hist[:most_likely_bin-int(std_bin/2)] )
	else:
		conf = sum( hist[:most_likely_bin-int(std_bin/2)] ) + sum( hist[most_likely_bin+int(std_bin/2):] )
	conf = 1 - conf
	#plt.stem(conf)
	#plt.show()

	if conf > 0.6:
		if mean_fc < (0.9 * SR/2):
			binary_result = True
		else:
			binary_result = False
	else:
		binary_result = False

	return mean_fc, conf, binary_result

def compute_spectral_envelope(frame_fft:list, xticks:list, kind="linear"):
	"""Compute the spectral envelope through a peak interpolation method

	Args:
		frame_fft: (list) iterable with the mono samples of a frame
		xticks: (list) xticks of the frame
	
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
	xp, hp = get_peaks(frame_fft, xticks) #compute the values of the local maxima of the function
	xp = np.append(xp,xticks[-1]) ; xp = np.append(0,xp) #appending the first value and last value in xticks
	hp = np.append(hp,frame_fft[-1]) ; hp = np.append(frame_fft[0],hp) #appending the first and last value of the function
	return interp1d(xp,hp,kind=kind)(xticks) #interpolating and returning

def compute_confidence(audio:list, frame_size:int, hop_size:int, fc:float, SR:int):
	"""Computes the confidence of the result by calculation the spectral power in
		the part of the spectrum from fc to fs/2 and comparing it to the whole spectral density

	Args:
		audio: (list) iterable with the mono information of the audio file
		fc: (float) predicted cut frequency
		SR: (int/float) sample rate

	Returns:
		(float) confidence value
	"""

def modify_floor(sig:list, floor_db:float, log=False):
	
	if log:
		sig[sig < (max(sig) + floor_db)] = max(sig) + floor_db
		return sig
	else:
		floor = 10 ** (floor_db/20)
		sig[sig < (max(sig) * floor)] = max(sig) * floor
		return sig

def compute_histogram(idx_arr:list, f:list, mask = []):
	if len(mask) == 0:
		hist = np.zeros(len(f))
		for idx in idx_arr:
			hist[idx] += 1
		return hist
	else:
		hist = np.zeros(len(f))
		for idx, boolean in zip(idx_arr, mask):
			if boolean:
				hist[idx] += 1
		return hist

def energy_verification(frame_fft:list, fc_index:int):
    return sum(frame_fft[:fc_index]**2)/sum(frame_fft**2) > 0.7

def detectBW(fpath:str, frame_size:float, hop_size:float, floor_db:float, oversample_f:int):

	if os.path.splitext(fpath)[1] != ".wav": raise ValueError("file must be wav") #check if the file has a wav extension, else: raise error
	if not is_power2(oversample_f): raise ValueError("oversample factor can only be 1, 2 or 4") #check if the oversample factor is a power of two

	#audio loader returns x, sample_rate, number_channels, md5, bit_rate, codec, of which only the first 3 are needed
	audio, SR = estd.AudioLoader(filename = fpath)()[:2]

	if audio.shape[1] != 1: audio = (audio[:,0] + audio[:,1]) / 2 #if stereo: downmix to mono
	
	frame_size *= oversample_f #if an oversample factor is desired, apply it
	f = np.arange(int(frame_size / 2) + 1)/frame_size * SR #initialize frequency vector or xticks
	
	energy_arr = []
	fc_index_arr = []
	interpolated_spectrum = np.zeros(int(frame_size / 2) + 1) #initialize interpolated_spectrum array
	fft = estd.FFT(size = frame_size) #declare FFT function
	window = estd.Windowing(size=frame_size, type="hann") #declare windowing function

	for i,frame in enumerate(estd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)):
		
		frame = window(frame) #apply window to the frame
		frame_fft = abs(fft(frame))
		frame_fft_db = 20 * np.log10(frame_fft + eps) #calculate frame fft values in db
		#energy_arr.append(energy(frame_fft))
		interp_frame = compute_spectral_envelope(frame_fft_db, f, "linear") #compute the linear interpolation between the values of the maxima of the spectrum
		interp_frame = modify_floor(interp_frame, floor_db, log=True)

		fc_index = compute_fc(interp_frame)
		fc_index_arr.append(fc_index)

		if energy_verification(frame_fft, fc_index):
			fc_index_arr.append(fc_index)
		else:
			fc_index_arr.append(len(f)-1)

		interpolated_spectrum += interp_frame #append the values to window
	
	interpolated_spectrum /= i + 1

	#energy_arr = normalise(energy_arr)
	#energy_mask = energy_arr>0.05

	hist = compute_histogram(fc_index_arr, f)
	fc, conf, binary = compute_mean_fc(hist, fc_index_arr, f, SR)

	print("filename: ", fpath ,"mean_fc: ", fc ," conf: ", conf ," binary_result: ", binary)

	#fig, ax = plt.subplots(4,1,figsize=(15,9))
	#ax[0].plot(fc_index_arr,"x")
	#ax[1].stem(f,hist)
	#ax[2].plot(energy_arr,'b')
	#ax[3].plot(f, interpolated_spectrum)
	#ax[3].axvline(x=fc,color="r")
	#plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calculates the effective BW of a file")
	parser.add_argument("fpath", help="relative path to the file")
	parser.add_argument("--frame_size", help="frame_size for the analysis fft (default=256)",default=256,required=False)
	parser.add_argument("--hop_size", help="hop_size for the analysis fft (default=128)",default=128,required=False)
	parser.add_argument("--floor_db", help="db value that will be considered as -inf",default=-90,required=False)
	parser.add_argument("--oversample", help="(int) factor for the oversampling in frequency domain. Must be a power of 2",default=1,required=False)
	args = parser.parse_args()
	detectBW(args.fpath, args.frame_size, args.hop_size, args.floor_db, int(args.oversample))

#python3 test2.py ../Dataset/Error_control_env/11k_in_44.1k.wav 
#python3 test2.py ../Dataset/Error_control_env/8KHz_in_44.1kHz.wav 
