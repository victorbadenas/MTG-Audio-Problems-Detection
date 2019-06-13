import argparse
from essentia.standard import AudioLoader
import numpy as np
import matplotlib.pyplot as plt

def autocorr(x, mode="half"):
	result = np.correlate(x, x, mode='full')
	if mode == "half":
		return result[int(result.size/2):]
	elif mode == "centered":
		return np.concatenate((result[:int(result.size/2)],result[int(result.size/2):]))
	else:
		return result

def main(path, frameSize=1024, hopSize=512):
	mu, sigma = 0, 1 # mean and standard deviation
	whiteNoise = np.random.normal(mu, sigma, 1024)
	noiseCorrelation = autocorr(whiteNoise, mode="centered")
	
	audio, _, channels, _, _, _ = AudioLoader(filename=path)()
	audio = np.sum(audio, axis=1)/channels
	frames = [audio[n * hopSize:min(len(audio), n*hopSize+frameSize)] for n in range(int(len(audio) / hopSize + 1))]
	bestFrame = np.argmax([np.sqrt(sum(frame**2)) for frame in frames])
	audio = frames[bestFrame]
	signalCorrelation = autocorr(audio, mode="centered")

	fig, ax = plt.subplots(2, 1, figsize=(16,9))
	ax[0].stem(noiseCorrelation/max(abs(noiseCorrelation)))
	ax[1].stem(signalCorrelation/max(abs(signalCorrelation)))
	plt.savefig("autocorrelation_comparison.png")
	plt.show()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Plots the results in the file")
    parser.add_argument("path", help="relative path to the tsv") # /Dataset/test/34120253.wav
    args = parser.parse_args()
    main(args.path)