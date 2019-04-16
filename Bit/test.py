import essentia.standard as estd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from essentia import array as esarr


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

	tuplelist = [(a, b) for a, b in zip(sig[peaks], xticks[peaks])]
	tuplelist.sort(key=lambda x: x[1])

	yval = [a for a, b in tuplelist]
	xval = [b for a, b in tuplelist]

	return xval, yval

def compute_histogram(sig:list, hist_xticks:list):


	hist = np.zeros(len(hist_xticks))
	for val in sig:
		hist[int(val)] += 1
	return hist

def Bit_Detection(fpath:str):

	if os.path.splitext(fpath)[1] != ".wav":
		# check if the file has a wav extension, else: raise error
		raise ValueError("file must be wav")

	#audio loader returns x, sample_rate, number_channels, md5, bit_rate, codec, of which only the first 3 are needed
	audio, SR, channels, _, br, _ = estd.AudioLoader(filename=fpath)()

	b = int(br / SR / channels) #number of bits used to code the fpath signal
	possible_values = np.arange(2 ** b)
	#_, ax = plt.subplots(3, channels, figsize=(15, 9))
	print("array created")

	#bits_result = -1
	#conf_result = 1
	audio_int_channel = (2**b) * ((0.5*audio[:, 0])+0.5)
	audio_int_channel = audio_int_channel[:10000]
	hist = compute_histogram(audio_int_channel, possible_values)
	plt.plot(hist, 'x')
	plt.show()
	assert False

	for channel in range(channels):
		audio_int_channel = (2**b) * ((0.5*audio[:, channel])+0.5)
		
		hist = compute_histogram(audio_int_channel, possible_values)

		#hist_peaks = hist/sum(hist)
		#hist_peaks[hist_peaks <= 0.0001] = 0
		#x_peaks, y_peaks = get_peaks(hist_peaks, possible_values)
		#y_peaks = np.array(y_peaks) * sum(hist)

		#tol = b/2
		#resolution = 2
		#center_x = []
		#center_y = []

		#first_idx = np.argmax(y_peaks) - resolution
		#for i in range(first_idx, first_idx + (3 * resolution + 1)):
		#	center_x.append(x_peaks[i])
		#	center_y.append(y_peaks[i])
		
		#b_pred = np.round(np.log2(np.mean(np.diff(center_x))))
		#b_pred = max(8,b_pred)
		#hop = 2 ** b_pred
		#print(hop)

		#zero_idx = 2 ** (b - 1)
		#idx_arr = []
		#idx = zero_idx - int(zero_idx/hop)*hop
		#while idx <= 2**b:
		#	idx_arr.append(idx)
		#	idx += hop

		#conf_hist = hist.copy()
		#for x_search in idx_arr:
		#	if (x_search - tol)<0:
		#		conf_hist[:int(x_search + tol)] = 0
		#	elif (x_search + tol)>len(conf_hist):
		#		conf_hist[int(x_search - tol):] = 0
		#	else:
		#		conf_hist[int(x_search - tol):int(x_search + tol)] = 0

		#print("b_pred: ", b_pred, "conf: ", 1-sum(conf_hist)/sum(hist))

		#bits_result = max(bits_result, b_pred)
		#conf_result *= 1-sum(conf_hist)/sum(hist)
		"""
		if channels == 1:
			ax[0].plot(audio_int_channel)
			ax[1].plot(possible_values, hist, 'x')
			ax[1].plot(x_peaks, y_peaks/sum(hist), 'x')
			ax[2].plot(possible_values, conf_hist, 'x')
		else:
			ax[0][channel].plot(audio_int_channel)
			ax[1][channel].plot(possible_values, hist, 'x')
			ax[1][channel].plot(x_peaks, y_peaks, 'xr')
			ax[1][channel].plot(x_peaks[np.argmax(y_peaks)], max(y_peaks), 'x')
			ax[2][channel].plot(possible_values, conf_hist, 'x')
		"""
	#print("bits_result: ", bits_result, "conf_result: ", conf_result)
	plt.plot(hist, 'x')
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="calculate correlation for all the sounds in s folder")
	parser.add_argument("directory", help="Directory of the files")
	args = parser.parse_args()
	Bit_Detection(args.directory)
