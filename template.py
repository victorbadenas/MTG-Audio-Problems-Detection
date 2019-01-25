import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import wavfile

def detectBW(directory):

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
		file = directory + "/" + file
		monoflag = False;

		if(not os.path.isfile(file)):
			continue

		try:
			fs,x = wavfile.read(file, mmap=True)
			print(file)

			samples = x.shape[0]
			channels = x.shape[1]
			monoflag = channels == 1

		except:
			continue

		soundcount += 1;
		print("Progress:" + "%.2f" % round(soundcount*100/numberofsounds,2))
		
	fid.write("\t\t</table>\n")
	fid.write("\t</body>\n")
	fid.write("</html>\n")

	fid.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="calculate correlation for all the sounds in s folder")
	parser.add_argument("directory", help="Directory of the files")
	args = parser.parse_args()
	detectBW(args.directory)