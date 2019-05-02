from scipy.interpolate import interp1d
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import essentia.standard as estd
import scipy.signal
from essentia import array as esarr
from tqdm import tqdm
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

def main(fpath: str, frame_size: float, hop_size: float):

	if os.path.splitext(fpath)[1] != ".wav":
		# check if the file has a wav extension, else: raise error
		raise ValueError("file must be wav")

	#audio loader returns x, sample_rate, number_channels, md5, bit_rate, codec, of which only the first 3 are needed
	audio, _ = estd.AudioLoader(filename=fpath)()[:2]

	if audio.shape[1] != 1:
		audio = (audio[:, 0] + audio[:, 1]) / 2  # if stereo: downmix to mono

	#fft = estd.FFT(size=frame_size)
	#window = estd.Windowing(size=frame_size, type="hann")
	arr = []
	for frame in tqdm(estd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)):
		frame = frame / max(frame) / frame_size
		ac = autocorr(frame)
		ac = ac / max(ac)
		#plt.plot(ac)
		#plt.show()
		arr.append(sum(abs(np.diff(ac))))
    
	#arr /= max(arr)
	#arr_env = compute_envelope(arr, np.arange(len(arr)))
	fig, ax = plt.subplots(2, 1, figsize=(15, 9))
	ax[0].plot(audio)
	#ax[1].plot(arr_env)
	ax[1].plot(arr)
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Calculates the effective BW of a file")
	parser.add_argument("fpath", help="relative path to the file")
	parser.add_argument(
		"--frame_size", help="frame_size for the analysis fft (default=256)", default=256, required=False)
	parser.add_argument(
		"--hop_size", help="hop_size for the analysis fft (default=128)", default=128, required=False)
	args = parser.parse_args()
	main(args.fpath, args.frame_size, args.hop_size)
