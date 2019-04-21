import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import wavfile

def calculateCorrelation(directory, upth, downth, M = 128):
	H = M
	count = np.array([0,0,0]) # [out of phase, normal, fake stereo]

	# fid = open(os.path.join(directory, 'Correlation.txt'), 'w')
	fid = open(os.path.join(directory, 'Correlation.html'), 'w')
	fid.write("<!DOCTYPE html>\n")
	fid.write("<html>\n")
	fid.write("\t<head></head>\n")
	fid.write("\t<body>\n")
	fid.write("\t\t<table style=\"width:100%\">\n")
	fid.write("\t\t<tr>\n")
	fid.write("\t\t\t<td><b>Name</b></td>\n")
	fid.write("\t\t\t<td><b>Correlation</b></td>\n")
	fid.write("\t\t\t<td><b>Label</b></td>\n")
	fid.write("\t\t</tr>\n")

	numberofsounds = []
	numberofsounds = np.sum([np.append(numberofsounds,1) for file in os.listdir(directory) if os.path.isfile(directory + "/" + file)])
	# numberofsounds = np.sum(numberofsounds)

	soundcount = 0
	for file in os.listdir(directory):
		file = directory + "/" + file

		if(not os.path.isfile(file)):
			continue

		soundcount += 1
		#with scipy.io.wavfile
		try:
			# print(file)
			fs,x = wavfile.read(file, mmap=True)

			if(len(x.shape) == 1):
				print("skipping " + file + " file as it is mono")
				continue

			samples = x.shape[0]
			channels = x.shape[1]

			nframes = int(np.ceil(samples/float(M)))
			# print(str(nframes*M) + " BUT THE FILE IS:" + str(samples))
			zpsize = int(nframes*M - samples)
			zp = np.zeros((zpsize,channels))
			x = np.concatenate((x,zp))


			# print("NOW IS: " + str(x.shape[0]))

			L = x[:,0]
			R = x[:,1]
				
		except:
			# print("ERROR IN :" + file)
			continue

		# plt.plot(L)
		# plt.plot(R)
		# plt.show()
		pearcorrarr = np.array([])
		fakezeros = 0
		for frame in range(1,nframes):
			Lframe = L[frame*H:frame*H+M-1]
			Rframe = R[frame*H:frame*H+M-1]
			cov = np.cov(Lframe,Rframe)[0][1];
			stdL = np.std(Lframe)
			stdR = np.std(Rframe)
			if(stdR*stdL == 0):
				pearcorrarr = np.append(pearcorrarr,0)
				fakezeros += 1
			else:
				pearcorrarr = np.append(pearcorrarr,cov/(stdR*stdL))

		pearcorr = sum(pearcorrarr)/(len(pearcorrarr)-fakezeros)
		pearcorr = max(pearcorr,-1)
		pearcorr = min(pearcorr,1)
		# print(file)
		# plt.subplot(2,1,1)
		# plt.plot(L)
		# plt.plot(R)
		# plt.subplot(2,1,2)
		# plt.plot(pearcorrarr)
		# plt.show()
		filename = file.split('/')
		filename = filename[int(len(filename)-1)]
		if(pearcorr>upth):
			label = "fake stereo";
			# print("FOR " + file + " ---------> CORRELATION =" + str(pearcorr) + "--------> LABEL: " + label)
			# fid.write(file + " ---------> CORRELATION =" + "%.2f" % round(pearcorr,2) + "--------> LABEL: " + label + '\n')
			fid.write("\t\t<tr>\n")
			fid.write("\t\t\t<td>" + filename + "</td>\n")
			fid.write("\t\t\t<td>" + "%.2f" % round(pearcorr,2) + "</td>\n")
			fid.write("\t\t\t<td>" + label + "</td>\n")
			fid.write("\t\t</tr>\n")
			count[2] += 1
		elif(pearcorr<downth):
			label = "out of phase"
			# print("FOR " + file + " ---------> CORRELATION =" + str(pearcorr) + "--------> LABEL: " + label)
			# fid.write(file + " ---------> CORRELATION =" + "%.2f" % round(pearcorr,2) + "--------> LABEL: " + label + '\n')
			fid.write("\t\t<tr>\n")
			fid.write("\t\t\t<td>" + filename + "</td>\n")
			fid.write("\t\t\t<td>" + "%.2f" % round(pearcorr,2) + "</td>\n")
			fid.write("\t\t\t<td>" + label + "</td>\n")
			fid.write("\t\t</tr>\n")
			count[0] += 1
		else:
			# print("FOR " + file + " ---------> CORRELATION =" + str(pearcorr))
			# fid.write(file + " ---------> CORRELATION =" + "%.2f" % round(pearcorr,2) + '\n')
			label = ""
			fid.write("\t\t<tr>\n")
			fid.write("\t\t\t<td>" + filename + "</td>\n")
			fid.write("\t\t\t<td>" + "%.2f" % round(pearcorr,2) + "</td>\n")
			fid.write("\t\t\t<td>" + label + "</td>\n")
			fid.write("\t\t</tr>\n")
			count[1] += 1

		print("Progress:" + "%.2f" % round(soundcount*100/numberofsounds,2))

	print(count)
	total = np.sum(count)

	finalstr = "\n Out of " + "%.2f" % round(total,2) + " sounds: " + "%.2f" % round(count[0]*100/total,2) + "% were out of phase " + "%.2f" % round(count[2]*100/total,2) + "% were fake stereo"
		
	fid.write("\t\t</table>\n")
	fid.write("\t" + finalstr)
	fid.write("\t</body>\n")
	fid.write("</html>\n")

	fid.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="calculate correlation for all the sounds in s folder")
	parser.add_argument("directory", help="Directory of the files")
	parser.add_argument("upth",help="Threshold for the fake stereo")
	parser.add_argument("downth",help="Threshold for the inverted phase")
	args = parser.parse_args()
	calculateCorrelation(args.directory, float(args.upth), float(args.downth))



			



		
		
	    
	        
