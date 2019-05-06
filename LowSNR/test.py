from scipy.interpolate import interp1d
from scipy.special import entr
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import essentia.standard as estd
import scipy.signal
from essentia import array as esarr
eps = np.finfo("float").eps

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return np.concatenate((result[:int(result.size/2)],result[int(result.size/2):]))

def get_peaks(sig: list, xticks: list):
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

	# get the index values of the local maxima of sig
	peaks, _ = scipy.signal.find_peaks(sig)

	tuplelist = [(a, b) for a, b in zip(xticks[peaks], sig[peaks])]
	tuplelist.sort(key=lambda x: x[1], reverse=True)

	xval = [a for a, b in tuplelist]
	yval = [b for a, b in tuplelist]

	return xval, yval

def compute_envelope(x: list, xticks: list, kind="linear"):
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
	xp, hp = get_peaks(
	    x, xticks)  # compute the values of the local maxima of the function
	xp = np.append(xp, xticks[-1])
	xp = np.append(0, xp)  # appending the first value and last value in xticks
	hp = np.append(hp, x[-1])
	# appending the first and last value of the function
	hp = np.append(x[0], hp)
	return interp1d(xp, hp, kind=kind)(xticks)  # interpolating and returning

def compute_ac_diff_sum(frame: list, normalize=True):

	if max(abs(frame))!=0:
		frame = frame / max(abs(frame))
	frame /= len(frame)
	ac = autocorr(frame)
	if normalize:
	    ac /= max(ac)
	return sum(abs(np.diff(ac)))

def compute_entropy(frame: list, bit_depth: int):
	frame -= frame.min()
	#print(max(frame), min(frame))
	frame = frame.astype('int64')
	nbins = max(frame)+1 #2**bit_depth
	# count the number of occurrences for each unique integer between 0 and x.max()
	# in each row of x
	counts = np.bincount(frame, minlength=nbins)

	# divide by number of columns to get the probability of each unique value
	p = counts / float(len(frame))

	# compute Shannon entropy in bits
	if len(p)==0:
		return 0
	else:
		return np.sum(entr(p) / np.log2(len(p)))

def main(fpath: str, frame_size: float, hop_size: float, entropy_th: float):

	if os.path.splitext(fpath)[1] != ".wav":
		# check if the file has a wav extension, else: raise error
		raise ValueError("file must be wav")

	#audio loader returns x, sample_rate, number_channels, md5, bit_rate, codec, of which only the first 3 are needed
	audio, SR, channels, _, br, _ = estd.AudioLoader(filename=fpath)()

	b = int(br / SR / channels) #number of bits used to code the fpath signal

	if audio.shape[1] != 1:
		audio = audio[:, 0]  # if stereo: downmix to mono
	audio = audio.astype("float32") / max(audio.astype("float32"))
	b = min(b,16)
	audio = esarr(audio.astype("float16"))
	max_nrg = max([sum(frame**2) for frame in
                estd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)])
	#fft = estd.FFT(size=frame_size)
	#window = estd.Windowing(size=frame_size, type="hann")
	ac_arr = []
	ent_arr = []
	nrg_arr = []
	sig_pwr = 0
	noise_pwr = 0
	sig_cnt = 0
	noise_cnt = 0
	ac_th = 0.01
	for frame in estd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
		ac = autocorr(frame)
		ac = abs(ac)
		ac /= sum(ac)
		ac = ac[int(len(ac)/2)]
		nrg = sum(frame**2)
		ac_arr.append(ac)
		nrg_arr.append(nrg)
		if nrg < 0.1*max_nrg:
			noise_pwr += nrg**2
			noise_cnt += 1
		else:
			if ac < ac_th:
				sig_pwr += nrg**2
				sig_cnt += 1
			else:
				noise_pwr += nrg**2
				noise_cnt += 1
		#ac = compute_ac_diff_sum(frame)
		#plt.plot(frame); plt.show()
		"""
		nrg = sum(frame**2)
		frame_int = ((2**(b-1)) * frame).astype('int')
		ent = compute_entropy(frame_int, b)
		ent_arr.append(ent)
		if nrg < 0.1*max_nrg: 
			noise_pwr += nrg**2
			noise_cnt += 1
		else:
			if ent < entropy_th:
				sig_pwr += nrg**2
				sig_cnt += 1
			else:
				noise_pwr += nrg**2
				noise_cnt += 1
		"""
	if noise_cnt == 0:
		SNR = np.inf
	elif sig_cnt == 0:
		SNR = 10 * np.log10(eps)
	else:
		sig_pwr /= sig_cnt
		noise_pwr /= noise_cnt
		SNR = 10 * np.log10(sig_pwr/noise_pwr)
	print("SNR: ", SNR)
	print("sig: {}, noise: {}".format(sig_cnt, noise_cnt))
	#print("Max Ent: ", max(ent_arr))
	#arr /= max(arr)
	#arr_env = compute_envelope(arr, np.arange(len(arr)))
	fig, ax = plt.subplots(3, 1, figsize=(15, 9))
	ax[0].plot(audio)
	#ax[1].plot(arr_env)
	ax[1].plot(ac_arr)
	#ax[1].plot(ent_arr)
	#ax[1].hlines(entropy_th,xmin = 0, xmax = len(ent_arr))
	ax[2].plot(nrg_arr/max_nrg)
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Calculates the effective BW of a file")
	parser.add_argument("fpath", help="relative path to the file")
	parser.add_argument(
		"--frame_size", help="frame_size for the analysis fft (default=1024)", default=1024, required=False)
	parser.add_argument(
		"--hop_size", help="hop_size for the analysis fft (default=512)", default=512, required=False)
	parser.add_argument(
		"--entropy_th", help="entropy threshold for stochastic frame detection (default=0.5)", default=0.5, required=False)
	args = parser.parse_args()
	main(args.fpath, int(args.frame_size), int(args.hop_size), float(args.entropy_th))
