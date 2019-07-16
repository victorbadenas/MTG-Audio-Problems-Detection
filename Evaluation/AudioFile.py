import numpy as np
import soundfile as sf
from sox import file_info
from scipy.signal import get_window
import os

class AudioFile(np.ndarray):
    def __new__(cls, *args, **kwargs):
        if len(args) != 1:
            raise ValueError("AudioFile takes 1 input: np.array for data or tr with the file path")
        input = args[0]
        if isinstance(input, str):
            if os.path.exists(input):
                data, sampleRate = sf.read(input)
            else:
                raise ValueError("file to load does not exist")
            instance = np.asarray(data).view(cls)
            instance.wavFile = input
            instance.absWavFile = os.path.abspath(input)
            instance.sampleRate = sampleRate
            instance._set_attributes()
        elif isinstance(input, AudioFile):
            instance = input
        else:
            if isinstance(input, np.ndarray):
                instance = np.asarray(input).view(cls)
            else:
                instance = np.asarray([]).view(cls)
            for k,v in kwargs.items():
                if k == "sampleRate":
                    instance.sampleRate = v
                elif k == "fileName":
                    if os.path.splitext(v) == ".wav":
                        instance.wavFile = v
                    else:
                        raise ValueError("filename must be a .wav file")
                else:
                    raise ValueError("{} is not a valid keyword")
        return instance

    def __array_finalize__(self, obj):
        if obj is None: return
        if isinstance(obj, AudioFile):
            attr = obj.__dict__
        else:
            attr = {}
        self.__dict__.update(attr)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        f = {
            "reduce": ufunc.reduce,
            "accumulate": ufunc.accumulate,
            "reduceat": ufunc.reduceat,
            "outer": ufunc.outer,
            "at": ufunc.at,
            "__call__": ufunc,
        }
        output = AudioFile(f[method](*(i.view(np.ndarray) for i in inputs), **kwargs))
        output.__dict__ = self.__dict__
        return output
    
    def __array_wrap__(self, out_arr, context=None):
        return super(AudioFile, self).__array_wrap__(self, out_arr, context)

    def _set_attributes(self):
        self.sampleRate = int(file_info.sample_rate(self.wavFile))
        self.bitDepth = file_info.bitrate(self.wavFile)
        self.bitRate = int(self.bitDepth * self.sampleRate)
        self.channels = file_info.channels(self.wavFile)
        self.comments = file_info.comments(self.wavFile)
        self.duration = file_info.duration(self.wavFile)
        self.encoding = file_info.encoding(self.wavFile)
        self.fileType = file_info.file_type(self.wavFile)
        self.numSamples = file_info.num_samples(self.wavFile)
        self.silent = file_info.silent(self.wavFile)

    def save(self, fileName=None, update=False):
        if not fileName:
            fileName = self.wavFile
        sf.write(fileName, self, self.sampleRate)
        if update:
            self.wavFile = fileName
            self._set_attributes()
    
    def frameGenerator(self, hopSize=512, frameSize=1024, zp=False, window=None):
        """
        window options: `boxcar`, `triang`, `blackman`, `hamming`, `hann`, `bartlett`,
        `flattop`, `parzen`, `bohman`, `blackmanharris`, `nuttall`, `barthann`
        """
        if window:
            window = get_window(window,frameSize, fftbins=True)
        else:
            window = np.ones(frameSize)
        if zp:
            numFrames = int(np.ceil(self.shape[0] / (hopSize + 1)))
            zpSignal = np.concatenate((self,np.zeros(numFrames*hopSize-self.shape[0])))
        else:
            zpSignal = self
        for n in range(int(self.shape[0] / (hopSize + 1))):
            if zp:
                if self.channels == 1:
                    frame = np.array(zpSignal[int(n * hopSize):int(n*hopSize+frameSize)])
                    yield frame * window
                else:
                    frame = np.array([zpSignal[int(n * hopSize):int(n*hopSize+frameSize),:]])
                    yield frame * window
            else:
                if self.channels == 1:
                    frame = np.array(zpSignal[int(n * hopSize):int(min(self.shape[0], n*hopSize+frameSize))])
                    yield frame * window
                else:
                    frame = np.array([zpSignal[int(n * hopSize):int(min(self.shape[0], n*hopSize+frameSize)),:]])
                    yield frame * window

    def normalised(self):
        return self / max(abs(self))

    def get_attr(self):
        return self.__dict__


if __name__ == "__main__":
    audio = AudioFile("audio/0a4d3800.wav")
    print(audio)
    for k, v in audio.get_attr().items():
        print("{}: {}".format(k,v))
    norm = audio.normalised()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,1)
    ax[0].plot(audio)
    ax[1].plot(norm)
    plt.show()
    audio.save()
    for frame in audio[:4000].frameGenerator(hopSize=1024, frameSize=2048, zp=True, window="blackmanharris"):
        print(frame, frame.shape)
        plt.plot(frame)
        plt.show()
    
