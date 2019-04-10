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
	for i,item in enumerate(reversed(bool_list)):
		if not item: return len(bool_list)-1-i

def compute_fc(std_arr:list, mean_arr:list, f:list, th:float):
	"""Applies the threshold and computes the limit frequency

    Args:
        std_arr: numpy.array of the standard deviation of the bins
        mean_arr: numpy.array of the mean of the bins
		f: numpy.array of the xticks for the two lists above
		th: (float) threshold for the desicion

    Returns:
        least restrictive frequency of the two calculations
    """
	if not len(std_arr) == len(mean_arr) == len(f): raise ValueError("length of the vectors must be equal")	

	#create a temporal array with normalise(std_arr-min(std_arr)) where the values are constrained between 0 and 1,
	#thresholding and get the last value that salisfies the condition from the end.
	std_pos = get_last_true_index( normalise(std_arr-min(std_arr)) < th)
	mean_pos = get_last_true_index( normalise(mean_arr-min(mean_arr)) < th)

	return max(f[std_pos], f[mean_pos])

def compute_spectral_envelope(frame:list, f:list):
	fp, hp = get_peaks(frame, f)
	fp = np.append(fp,f[-1]) ; fp = np.append(0,fp)
	hp = np.append(hp,frame[-1]) ; hp = np.append(frame[0],hp)
	return interp1d(fp,hp,kind="linear")(f)

def compute_confidence(x:list, fc:float, SR:int):

	x = estd.Windowing(size=len(x), type="hann")(x)
	N = int(2**(np.ceil(np.log2(len(x)))))
	x = np.append(x,np.zeros(N-len(x)))
	x = esarr(x)
	tfX = abs(estd.FFT()(x))
	f = np.arange(int(len(x)/2)+1)/len(x)*SR
	fck = int(len(tfX) * fc/(0.5*SR))
	#plt.plot(f, tfX)
	#plt.axvline(x=fc,color="r")
	#plt.show()
	return 1-sum(tfX[fck:])/sum(tfX)

def detectBW(fpath:str, frame_size:float, hop_size:float, floor_db:float, th:float, oversample_f:int):

	if os.path.splitext(fpath)[1] != ".wav": raise ValueError("file must be wav") #check if the file has a wav extension, else: raise error
	if not is_power2(oversample_f): raise ValueError("oversample factor can only be 1, 2 or 4") #check if the oversample factor is a power of two

	#audio loader returns x, sample_rate, number_channels, md5, bit_rate, codec, of which only the first 3 are needed
	x, SR = estd.AudioLoader(filename = fpath)()[:2]
	print(x.shape,SR) 

	if x.shape[1] != 1: x = (x[:,0] + x[:,1]) / 2 #if stereo: downmix to mono
	
	frame_size *= oversample_f #if an oversample factor is desired, 
	f = np.arange(int(frame_size/2)+1)/frame_size * SR #initialize frequency vector or xticks

	interpolated_signals = [] #initialize interpolated_signals array
	fft = estd.FFT(size = frame_size) #declare FFT function
	window = estd.Windowing(size=frame_size, type="hann") #declare windowing function

	for frame in estd.FrameGenerator(x, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
		
		frame = window(frame) #apply window to the frame
		frame_fft_db = 20 * np.log10(abs(fft(frame) + eps)) #calculate frame fft values in db
		#each value less than the threshold is set to 30 dB lower than the threshold
		print(max(frame_fft_db)-120)
		frame_fft_db[ frame_fft_db < (max(frame_fft_db) + floor_db)] = max(frame_fft_db) + floor_db
		interp_frame = compute_spectral_envelope(frame_fft_db, f) #compute the linear interpolation between the values of the maxima of the spectrum
		interpolated_signals.append(interp_frame) #append the values to window
	
	std_arr = np.std(np.array(interpolated_signals), axis = 0) #calculate the stardard deviation of each bin frequency in the interpolated spectrums in db
	mean_arr = np.mean((interpolated_signals), axis = 0) #calculate the mean of each bin frequency in the interpolated spectrums in db

	std_arr = normalise(std_arr, log=False)
	mean_arr = normalise(mean_arr, log=True)

	fc = compute_fc(std_arr, mean_arr, f, th) #apply the threshold and find the cut frequency
	confidence = compute_confidence(x,fc,SR) #calculate the confidence of the algorithm for the predicted fc

	print("fc:", fc, "confidence:", confidence)

	#plot of the std and mean plots.
	fig,ax = plt.subplots(2,1,figsize=(10,8))
	ax[0].plot(f,mean_arr)
	ax[0].set_title("mean")
	ax[0].axvline(x=fc,color="r")
	ax[1].plot(f,std_arr)
	ax[1].set_title("standard deviation")
	ax[1].axvline(x=fc,color="r")
	plt.show()
	#plt.clf()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calculates the effective BW of a file")
	parser.add_argument("fpath", help="relative path to the file")
	parser.add_argument("--frame_size", help="frame_size for the analysis fft (default=256)",default=256,required=False)
	parser.add_argument("--hop_size", help="hop_size for the analysis fft (default=128)",default=128,required=False)
	parser.add_argument("--floor_db", help="db value that will be considered as -inf",default=-60,required=False)
	parser.add_argument("--th", help="threshold for the standard deviation to be considered in the detection process linear [0,1]",default=0.4,required=False)
	parser.add_argument("--oversample", help="(int) factor for the oversampling in frequency domain. Must be a powerr of 2",default=1,required=False)
	args = parser.parse_args()
	detectBW(args.fpath, args.frame_size, args.hop_size, args.floor_db, args.th, int(args.oversample))

#python3 test.py ../Dataset/BW\ detection/_m1_DistNT_65.wav
#python3 test.py ../Dataset/BW\ detection/Door\ of\ flat\ close\ int\ block\ of\ flats.wav
#python3 test.py ../Dataset/BW\ detection/Door\ open\ close\ int\ flat\ soft\ 2.wav