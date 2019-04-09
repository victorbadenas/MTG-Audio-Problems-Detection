import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import essentia.standard as estd
from scipy.signal import hilbert, find_peaks, get_window

from numpy import array, sign, zeros
from scipy.interpolate import interp1d

def normalize(sig:list, log=False):
	return sig / max(sig) if not log else (sig - max(sig))

def smooth_function(sig, window_type= 'triang', window_len = 50, mode = 'same'):
    return np.convolve(sig, get_window(window_type, window_len), mode=mode)

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

def findEnvelopes(s):
	
	q_u = zeros(s.shape)
	q_l = zeros(s.shape)

	#Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.

	u_x = [0,]
	u_y = [s[0],]

	l_x = [0,]
	l_y = [s[0],]

	#Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

	for k in range(1,len(s)-1):
	    if (sign(s[k]-s[k-1])==1) and (sign(s[k]-s[k+1])==1):
	        u_x.append(k)
	        u_y.append(s[k])

	    if (sign(s[k]-s[k-1])==-1) and ((sign(s[k]-s[k+1]))==-1):
	        l_x.append(k)
	        l_y.append(s[k])

	#Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.

	u_x.append(len(s)-1)
	u_y.append(s[-1])

	l_x.append(len(s)-1)
	l_y.append(s[-1])

	#Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.

	u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
	l_p = interp1d(l_x,l_y,kind = 'cubic',bounds_error = False, fill_value=0.0)

	#Evaluate each model over the domain of (s)
	for k in range(0,len(s)):
	    q_u[k] = u_p(k)
	    q_l[k] = l_p(k)

	#Return everything
	return q_u,q_l

def detectBW(fpath, frame_size=1024, hop_size=512):

	relativeBW = -1

	_, extension = os.path.splitext(fpath)
	if extension != ".wav": raise ValueError("file must be wav")

	print(fpath)
	x, SR, channels, _, br, _ = estd.AudioLoader(fpath)()

	print(x.shape, SR, channels, br)

	#if channels != 1:
	#	x = (x[:,0] + x[:,1])/2
	
	

	
	#plt.plot(f,mX[:int(N/2)])
	#plt.plot(f,smX[:int(N/2)],color='r')
	#plt.savefig(fname+".png")
	#plt.clf()

	#print("Progress:" + "%.2f" % round((i+1)*100/numberoffiles,2) + " %")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calculates the effective BW of a file")
	parser.add_argument("fpath", help="relative path to the file")
	parser.add_argument("--smoothingfactor", help="how many times the upper envelope is computed in itself (>1)",default=1,required=False)
	args = parser.parse_args()
	detectBW(args.fpath)
