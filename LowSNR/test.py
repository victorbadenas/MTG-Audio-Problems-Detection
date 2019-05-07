from scipy.special import entr
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import essentia.standard as estd
from essentia import array as esarr
eps = np.finfo("float").eps

def autocorr(x, mode = "half"):
	result = np.correlate(x, x, mode='full')
	if mode == "half":
		return result[int(result.size/2):]
	elif mode == "centered":
		return np.concatenate((result[:int(result.size/2)],result[int(result.size/2):]))
	else:
		return result

def compute_ac_diff_sum(frame: list, normalize=True):

	if max(abs(frame))!=0:
		frame = frame / max(abs(frame))
	frame /= len(frame)
	ac = autocorr(frame)
	if normalize:
	    ac /= max(ac)
	return sum(abs(np.diff(ac)))

def compute_entropy(frame: list, bit_depth: int):
	frame -= frame.min()
	#print(max(frame), min(frame))
	frame = frame.astype('int64')
	nbins = max(frame)+1 #2**bit_depth
	# count the number of occurrences for each unique integer between 0 and x.max()
	# in each row of x
	counts = np.bincount(frame, minlength=nbins)

	# divide by number of columns to get the probability of each unique value
	p = counts / float(len(frame))

	# compute Shannon entropy in bits
	if len(p)==0:
		return 0
	else:
		return np.sum(entr(p) / np.log2(len(p)))

def main(fpath: str, frame_size: float, hop_size: float, entropy_th: float):

	if os.path.splitext(fpath)[1] != ".wav":
		# check if the file has a wav extension, else: raise error
		raise ValueError("file must be wav")

	#audio loader returns x, sample_rate, number_channels, md5, bit_rate, codec, of which only the first 3 are needed
	audio, _, _, _, _, _ = estd.AudioLoader(filename=fpath)()

	#bit_depth = int(br / SR / channels) #number of bits used to code the fpath signal
	if audio.shape[1] > 1:
		audio = np.reshape(audio, audio.shape[0]*audio.shape[1], order='F')
	audio = audio.astype("float32") / max(audio.astype("float32"))
	#bit_depth = min(bit_depth,16)
	audio = esarr(audio.astype("float16"))
	max_nrg = max([sum(frame**2) for frame in
                estd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)])
	ac_arr = []
	nrg_arr = []
	sig_pwr = 0
	noise_pwr = 0
	sig_cnt = 0
	noise_cnt = 0
	ac_th = 0.6

	for frame in estd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
		ac = abs(autocorr(frame, mode="half"))
		#ac /= sum(ac)
		#plt.plot(ac); plt.show()
		nrg = sum(frame**2)
		ac = ac[0]/sum(ac) if sum(ac)>0 else 0
		nrg = nrg/max_nrg if max_nrg>0 else 0
		ac_arr.append(ac)
		nrg_arr.append(nrg)

	ac_arr /= max(ac_arr)
	for nrg, ac in zip(nrg_arr, ac_arr):
		if nrg < 0.1:
			noise_pwr += nrg**2
			noise_cnt += 1
		else:
			if ac < ac_th:
				sig_pwr += nrg**2
				sig_cnt += 1
			else:
				noise_pwr += nrg**2
				noise_cnt += 1
	
	if noise_cnt == 0:
		SNR = np.inf
	elif sig_cnt == 0:
		SNR = 10 * np.log10(eps)
	else:
		sig_pwr /= sig_cnt
		noise_pwr /= noise_cnt
		SNR = 10 * np.log10(sig_pwr/noise_pwr)
	
	print("SNR: ", SNR)
	print("sig: {}, noise: {}".format(sig_cnt, noise_cnt))
	print("conf: ", 1-abs(noise_cnt-sig_cnt)/(sig_cnt + noise_cnt))
	#print("Max Ent: ", max(ent_arr))
	#arr /= max(arr)
	#arr_env = compute_envelope(arr, np.arange(len(arr)))
	_, ax = plt.subplots(3, 1, figsize=(15, 9))
	ax[0].plot(audio)
	#ax[1].plot(arr_env)
	ax[1].plot(ac_arr)
	#ax[1].plot(ent_arr)
	#ax[1].hlines(entropy_th,xmin = 0, xmax = len(ent_arr))
	ax[2].plot(nrg_arr)
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Calculates the effective BW of a file")
	parser.add_argument("fpath", help="relative path to the file")
	parser.add_argument(
		"--frame_size", help="frame_size for the analysis fft (default=1024)", default=1024, required=False)
	parser.add_argument(
		"--hop_size", help="hop_size for the analysis fft (default=512)", default=512, required=False)
	parser.add_argument(
		"--entropy_th", help="entropy threshold for stochastic frame detection (default=0.5)", default=0.5, required=False)
	args = parser.parse_args()
	main(args.fpath, int(args.frame_size), int(args.hop_size), float(args.entropy_th))