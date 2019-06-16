from numpy.random import normal
from numpy import log10, finfo, arange
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal.windows import get_window
from scipy.signal import decimate, resample
eps = finfo("float").eps

def normalise(x):
	return (x-min(x))/(max(x)-min(x))

sr = 44100
mean = 0
std = 1 
N = 2**14
noise = normal(mean, std, size=N)
noisedec = resample(decimate(noise, 2), N)
window = get_window('blackmanharris', N, fftbins=False)

noisefft = abs(fft(noise * window))[:int(N/2)]
noisefftdec = abs(fft(noisedec * window))[:int(N/2)]
noisefft = 20*log10(normalise(noisefft) + eps)
noisefftdec = 20*log10(normalise(noisefftdec) + eps)

f = arange(int(N/2))/N*sr

fig, ax = plt.subplots(2, 1, figsize=(16,9))
ax[0].plot(f,noisefft)
ax[1].plot(f,noisefftdec)
plt.show()