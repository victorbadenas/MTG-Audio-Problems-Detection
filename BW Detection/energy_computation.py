import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import essentia.standard as estd
import scipy.signal
from scipy.interpolate import interp1d
from essentia import array as esarr
eps = np.finfo("float").eps

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

def compute_mean_fc(fft_lin:list, fc_index_arr:list, xticks:list, SR:float, hist=[]):
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
	if len(hist)==0:
		hist = compute_histogram(fc_index_arr, xticks) #computation of the histogram
	
	hist[hist<.1*max(hist)]=0
	most_likely_bin = np.argmax(hist) #bin value of the highest peak of the histogram
	#most_likely_bin = get_last_true_index(hist != 0)
	#the confidence value changes depending on if most_likely_bin falls under the 85% lower spectrogram or not
	if most_likely_bin <= .85*len(hist):
		#if it falls under the 85% lower, the confidence is computed by a weighted sum of the values of the histogram,
		#the highest peak having the highest importance and decreasing as the indexes go further.
		"""
		#creation of the confidence scale
		conf_scale = abs(most_likely_bin - np.arange(len(hist))); conf_scale = max(conf_scale) - conf_scale ; conf_scale = conf_scale / max(conf_scale)
		conf = sum(hist * conf_scale) / sum(hist) #computation of the confidence sum, normalised by the histogram length
		
		conf_scale = abs(most_likely_bin - np.arange(len(hist))); conf_scale = conf_scale / max(conf_scale); conf_scale = conf_scale**(1/5)
		plt.plot(conf_scale); plt.show()
		conf = 1 - sum(hist * conf_scale) / sum(hist)"""
		conf = sum(hist[most_likely_bin-1:most_likely_bin+1]) / sum(hist[:most_likely_bin+1])
		#conf = sum(hist[int(most_likely_bin-int(0.016*len(hist))):int(most_likely_bin+int(0.016*len(hist)))])/sum(hist)
		#return the analog frequency corresponding to the bin, confidence value, and True if the confidence value is higher than 0.77
		return most_likely_bin, conf, conf>0.8
	else:
		#if it falls over the 85% mark, the confidence is computated by summing the square of the 3 samples of the histogram closer to the max
		#and compare it to the sum of all the values appended to the histogram.
		conf = sum(hist[int(.85*len(hist)):]) / sum(hist)
		return most_likely_bin, conf, False

def detectBW(fpath:str, frame_size:float, hop_size:float, floor_db:float, oversample_f:int):
	
	if os.path.splitext(fpath)[1] != ".wav": raise ValueError("file must be wav") #check if the file has a wav extension, else: raise error
	if not is_power2(oversample_f): raise ValueError("oversample factor can only be 1, 2 or 4") #check if the oversample factor is a power of two

	#audio loader returns x, sample_rate, number_channels, md5, bit_rate, codec, of which only the first 3 are needed
	audio, SR = estd.AudioLoader(filename = fpath)()[:2]

	if audio.shape[1] != 1: audio = (audio[:,0] + audio[:,1]) / 2 #if stereo: downmix to mono
	
	frame_size *= oversample_f #if an oversample factor is desired, apply it

	fc_index_arr = []
	hist = np.zeros(int(frame_size/2+1))
	fft = estd.FFT(size = frame_size) #declare FFT function
	window = estd.Windowing(size=frame_size, type="hann") #declare windowing function
	avg_frames = np.zeros(int(frame_size/2)+1)

	max_nrg = max([sum(abs(fft(window(frame)))**2) for frame in 
				estd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)])

	for i,frame in enumerate(estd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)):
		
		frame = window(frame) #apply window to the frame
		frame_fft = abs(fft(frame))
		nrg = sum(frame_fft**2)

		if nrg >= 0.1*max_nrg:
			for i in reversed(range(len(frame_fft))):
				if sum(frame_fft[i:]/i) >= 1e-5:
					fc_index_arr.append(i)
					hist[i] += nrg
					break
			avg_frames = avg_frames + frame_fft
	if len(fc_index_arr)==0: 
		fc_index_arr.append(int(frame_size/2)+1)
		hist[int(frame_size/2)] += 1
	
	avg_frames /= (i+1)
	most_likely_bin, conf, binary = compute_mean_fc(avg_frames, fc_index_arr, np.arange(int(frame_size/2)+2), SR, hist=hist)

	print("f={:0=2f}, conf={:0=2f}, problem={}".format(
		most_likely_bin*SR / frame_size, conf, str(binary)))
	fig, ax = plt.subplots(2,1,figsize=(15,9))
	ax[0].plot(20 * np.log10(avg_frames + eps))
	ax[0].axvline(x=most_likely_bin, color = 'r')
	ax[0].set_ylim(bottom=-120)
	ax[1].stem(hist)
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calculates the effective BW of a file")
	parser.add_argument("fpath", help="relative path to the file")
	parser.add_argument("--frame_size", help="frame_size for the analysis fft (default=256)",default=256,required=False)
	parser.add_argument("--hop_size", help="hop_size for the analysis fft (default=128)",default=128,required=False)
	parser.add_argument("--floor_db", help="db value that will be considered as -inf",default=-90,required=False)
	parser.add_argument("--oversample", help="(int) factor for the oversampling in frequency domain. Must be a power of 2",default=1,required=False)
	args = parser.parse_args()
	detectBW(args.fpath, args.frame_size, args.hop_size, args.floor_db, int(args.oversample))
