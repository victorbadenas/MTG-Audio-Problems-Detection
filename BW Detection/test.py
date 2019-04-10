import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import essentia.standard as estd
from scipy.signal import hilbert, find_peaks, get_window
from sklearn import linear_model
eps = np.finfo("float").eps

from numpy import array, sign, zeros
from scipy.interpolate import interp1d

def normalise(sig:list, log=False):
	return sig / max(sig) if not log else (sig - max(sig))

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

    peaks, _ = find_peaks(sig)

    tuplelist = [(a, b) for a, b in zip(xticks[peaks], sig[peaks])]
    tuplelist.sort(key=lambda x: x[1], reverse=True)

    xval = [a for a, b in tuplelist]
    yval = [b for a, b in tuplelist]

    return xval, yval

def get_last_true_index(bool_list:list):
	for i,item in enumerate(reversed(bool_list)):
		if not item: return len(bool_list)-1-i

def compute_fc(std_arr:list,mean_arr:list,f:list, std_th:float, mean_th:float):
	if not len(std_arr) == len(mean_arr) == len(f): raise ValueError("length of the vectors must be equal")	
	std_pos = get_last_true_index(std_arr<std_th)
	mean_pos = get_last_true_index(mean_arr<mean_th)
	print(std_pos, mean_pos)
	print(f[std_pos], f[mean_pos])
	return max(f[std_pos], f[mean_pos])

def detectBW(fpath, frame_size=256, hop_size=128, floor_db=-30, std_th=0.65, mean_th=-15):

	_, extension = os.path.splitext(fpath)
	if extension != ".wav": raise ValueError("file must be wav")

	print(fpath)
	x, SR, channels, _, br, _ = estd.AudioLoader(filename = fpath)()

	if channels != 1: x = (x[:,0] + x[:,1]) / 2
	
	print(x.shape, SR, channels, br)
	
	effectiveBW = []
	window = estd.Windowing(size=frame_size, type="hann")

	interpolated_signals = []
	f = np.arange(int(frame_size/2)+1)/frame_size * SR
	fft = estd.FFT(size = frame_size)

	for frame in estd.FrameGenerator(x, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
		
		frame = window(frame)
		frame_fft = fft(frame) + eps
		frame_fft_db = 20 * np.log10(abs(frame_fft))
		floor_db_relative = max(frame_fft_db) + floor_db
		frame_fft_db[frame_fft_db<floor_db_relative] = -120
		
		xp, yp = get_peaks(frame_fft_db, f)
		xp = np.append(xp,f[-1])
		xp = np.append(0,xp)
		yp = np.append(yp,frame_fft_db[-1])
		yp = np.append(frame_fft_db[0],yp)
		
		interp_func = interp1d(xp,yp,kind="linear")
		interp_frame = interp_func(f)

		interpolated_signals.append(interp_frame)
	
	std_arr = np.std(np.array(interpolated_signals), axis = 0)
	mean_arr = np.mean((interpolated_signals), axis = 0)

	std_arr = normalise(std_arr, log=False)
	mean_arr = normalise(mean_arr, log=True)

	fc = compute_fc(std_arr[1:],mean_arr[1:],f[1:], std_th, mean_th)

	fig,ax = plt.subplots(2,1,figsize=(10,8))
	ax[0].plot(f,mean_arr)
	#ax[0].semilogx(f[1:],mean_arr[1:])
	ax[0].set_title("mean")
	ax[0].axvline(x=fc,color="r")
	ax[1].plot(f,std_arr)
	#ax[1].semilogx(f[1:],std_arr[1:])
	ax[1].set_title("standard deviation")
	ax[1].axvline(x=fc,color="r")
	plt.show()
	#plt.clf()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calculates the effective BW of a file")
	parser.add_argument("fpath", help="relative path to the file")
	parser.add_argument("--smoothingfactor", help="how many times the upper envelope is computed in itself (>1)",default=1,required=False)
	args = parser.parse_args()
	detectBW(args.fpath)

#python3 test.py ../Dataset/BW\ detection/_m1_DistNT_65.wav
#python3 test.py ../Dataset/BW\ detection/Door\ of\ flat\ close\ int\ block\ of\ flats.wav
#python3 test.py ../Dataset/BW\ detection/Door\ open\ close\ int\ flat\ soft\ 2.wav