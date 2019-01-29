import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import wavfile
from scipy.signal import hilbert

from numpy import array, sign, zeros
from scipy.interpolate import interp1d

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

def detectBW(directory,sf=1):

	fid = open(os.path.join(directory, 'BW.html'), 'w')
	fid.write("<!DOCTYPE html>\n")
	fid.write("<html>\n")
	fid.write("\t<head></head>\n")
	fid.write("\t<body>\n")
	fid.write("\t\t<table style=\"width:100%\">\n")
	fid.write("\t\t<tr>\n")
	fid.write("\t\t\t<td><b>Name</b></td>\n")
	fid.write("\t\t\t<td><b>BW</b></td>\n")
	fid.write("\t\t\t<td><b>SR</b></td>\n")
	fid.write("\t\t</tr>\n")

	numberofsounds = []
	numberofsounds = np.sum([np.append(numberofsounds,1) for file in os.listdir(directory) if os.path.isfile(directory + "/" + file)])
	# numberofsounds = np.sum(numberofsounds)

	soundcount = 0
	for file in os.listdir(directory):
		relativeBW = -1
		SR = -1
		fname = file
		file = directory + "/" + file
		monoflag = False;

		if(not os.path.isfile(file)):
			continue

		try:
			soundcount += 1
			fs,x = wavfile.read(file, mmap=True)
			samples = x.shape[0]
			channels = x.shape[1]
			monoflag = channels == 1

			if not monoflag:
				x = (x[:,0] + x[:,1])/2

		except:
			continue

		N = int(pow(2, np.ceil(np.log2(len(x)))))
		x = np.floor(2*x)/2**15
		x = np.append(x,np.zeros(N-len(x)))
		X = np.fft.fft(x)/np.sum(x)
		mX = 20*np.log10(np.abs(X))

		hemX,_ = findEnvelopes(mX)
		hemX,_ = findEnvelopes(hemX)

		f = np.arange(int(N/2))/N*fs
		plt.plot(f,mX[:int(N/2)])
		plt.plot(f,hemX[:int(N/2)])
		plt.savefig(fname+".png")
		plt.clf()

		print("Progress:" + "%.2f" % round(soundcount*100/numberofsounds,2) + " %")
		
	fid.write("\t\t</table>\n")
	fid.write("\t</body>\n")
	fid.write("</html>\n")

	fid.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calculates the effective BW of a file")
	parser.add_argument("--directory", help="Directory of the files",default="./",required=False)
	parser.add_argument("--smoothingfactor", help="how many times the upper envelope is computed in itself (>1)",default=1,required=False)
	args = parser.parse_args()
	detectBW(args.directory)
