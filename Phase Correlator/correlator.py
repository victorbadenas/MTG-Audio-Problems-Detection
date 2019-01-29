import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import wavfile

def calculateCorrelation(directory, upth, downth):

	for file in os.listdir(directory):
		file = directory + "/" + file;

		if(not os.path.isfile(file)):
			continue

		try:
			fs,x = wavfile.read(file, mmap=True)
			try:
				L = x[:,0]
				R = x[:,1]
			except:
				L = x
				R = x
		except:
			print("ERROR IN :" + file)
			continue

		# plt.plot(L)
		# plt.plot(R)
		# plt.show()
		cov = np.cov(L,R)[0][1];
		stdL = np.std(L)
		stdR = np.std(R)
		pearcorr = max(-1.0,min(1.0,cov/(stdR*stdL)));
		if(pearcorr>upth):
			label = "mono";
			print("FOR " + file + " ---------> CORRELATION =" + str(pearcorr) + "--------> LABEL: " + label)
		elif(pearcorr<downth):
			label = "out of phase"
			print("FOR " + file + " ---------> CORRELATION =" + str(pearcorr) + "--------> LABEL: " + label)
		else:
			print("FOR " + file + " ---------> CORRELATION =" + str(pearcorr))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="calculate correlation for all the sounds in s folder")
	parser.add_argument("directory", help="Directory of the files")
	parser.add_argument("upth",help="Upper threshold for the fake stereo")
	parser.add_argument("downth",help="Lower threshold for the fake stereo")
	args = parser.parse_args()
	calculateCorrelation(args.directory, float(args.upth), float(args.downth))



		



	
	
    
        
