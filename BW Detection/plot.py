import essentia.standard as estd
from essentia import array as esarr
import matplotlib.pyplot as plt
import os
import numpy as np

DIR = "../Dataset/BW detection/"

for file in os.listdir(DIR):

    fpath = os.path.join(DIR,file)

    name, extension = os.path.splitext(file)
    print(file)
    if extension == ".wav":
        x, SR, channels, _, br, _ = estd.AudioLoader(filename = fpath)()
        

        channels = x.shape[1]
        if channels != 1: x = (x[:,0] + x[:,1]) / 2
        print(x.shape,SR,channels,br)

        window = estd.Windowing(size=len(x), type="hann")
        x = window(x)
        N = int(2**(np.ceil(np.log2(len(x)))))
        x = np.append(x,np.zeros(N-len(x)))
        x = esarr(x)
        tfX = estd.FFT()(x)
        tfX = 20*np.log10(abs(tfX))
        f = np.arange(int(len(x)/2)+1)/len(x)*SR
        plt.plot(f, tfX[:int(len(x)/2)+1])
        plt.savefig(os.path.join(DIR,name+".png"))
        plt.clf()
