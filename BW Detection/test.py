import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import wavfile
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

def detectBW(directory,sf=2):

	numberoffiles = np.sum([1 for file in os.listdir(directory)])
	# numberofsounds = np.sum(numberofsounds)

	for i,fname in enumerate(os.listdir(directory)):
		relativeBW = -1
		SR = -1
		fname = os.path.join(directory,fname)

		_, extension = os.path.splitext(fname)
		if extension != ".wav":
			continue

		fs,x = wavfile.read(fname, mmap=True)
		samples = x.shape[0]
		channels = x.shape[1] if len(x.shape) > 1 else 1

		if channels != 1:
			x = (x[:,0] + x[:,1])/2

		N = int(pow(2, np.ceil(np.log2(len(x)))))

		f = np.arange(int(N/2))/N*fs
		x = np.floor(2*x)/2**15
		x = np.append(x,np.zeros(N-len(x)))
		X = np.fft.fft(x)/np.sum(x)
		mX = 20*np.log10(np.abs(X))
		
		#mX = mX * np.log(np.arange(N))
		mX = normalize(mX,log=True)
		smX = mX
		for _ in range(sf):
			smX,_ = findEnvelopes(smX)
		
		#smX = smooth_function(smX, window_type= 'hann', window_len = 30)
		#smX = normalize(smX,log=True)
		#xpeaks, ypeaks = get_peaks(smX[:int(N/2)],f)

		
		plt.plot(f,mX[:int(N/2)])
		plt.plot(f,smX[:int(N/2)],color='r')
		plt.savefig(fname+".png")
		plt.clf()

		print("Progress:" + "%.2f" % round((i+1)*100/numberoffiles,2) + " %")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calculates the effective BW of a file")
	parser.add_argument("--directory", help="Directory of the files",default="./",required=False)
	parser.add_argument("--smoothingfactor", help="how many times the upper envelope is computed in itself (>1)",default=1,required=False)
	args = parser.parse_args()
	detectBW(args.directory)
