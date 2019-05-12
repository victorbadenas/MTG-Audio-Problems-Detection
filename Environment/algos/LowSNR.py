import essentia.standard as estd
import numpy as np
import argparse
from scipy.special import entr
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

def lowSNR_detector(audio:list, frame_size=1024, hop_size=512, nrg_th=0.1, ac_th=0.6, snr_th=5):

	if audio.shape[1] > 1:
		audio = np.reshape(audio, audio.shape[0]*audio.shape[1], order='F')
	
	audio = audio.astype("float32") / max(audio.astype("float32"))
	audio = esarr(audio.astype("float16"))
	ac_arr = []
	nrg_arr = []
	sig_pwr = 0
	noise_pwr = 0
	sig_cnt = 0
	noise_cnt = 0
	ac_th = 0.6

	for frame in estd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
		ac = abs(autocorr(frame, mode="half"))
		nrg = sum(frame**2)
		ac = ac[0]/sum(ac) if sum(ac)>0 else 0
		ac_arr.append(ac)
		nrg_arr.append(nrg)
	
	ac_arr /= max(ac_arr)
	nrg_arr /= max(nrg_arr)

	for nrg, ac in zip(nrg_arr, ac_arr):
		if nrg < nrg_th:
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
		snr = np.inf
	elif sig_cnt == 0:
		snr = 10 * np.log10(eps)
	else:
		sig_pwr /= sig_cnt
		noise_pwr /= noise_cnt
		snr = 10 * np.log10(sig_pwr/noise_pwr)

	#conf = 1-abs(noise_cnt-sig_cnt)/(sig_cnt + noise_cnt)
	#if conf > 0.7 and snr < snr_th:
	#	return snr, conf, True
	#return snr, conf, False

	return snr, snr < snr_th

	