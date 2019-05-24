import numpy as np
import matplotlib.pyplot as plt
eps = np.finfo("float").eps


class LowSnrDetector:
    def __init__(self, frameSize=1024, hopSize=512, nrgThreshold=0.1, acThreshold=0.6, snrThreshold=5, mode="average"):
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.nrgThreshold = nrgThreshold
        self.acThreshold = acThreshold
        self.snrThreshold = snrThreshold
        if mode in ["average", "accumulative"]:
            self.mode = mode
        else:
            raise ValueError("mode must be either accumulative or average")
        self._reset_()

    def __call__(self, audio):
        self._reset_()
        self.audio = audio
        self._concatenateChannels_()
        self._computeAcNrg_()
        self._normalise_()
        self._classifier_()
        self._compute_()
        return self.snr, self.snr < self.snrThreshold

    def _reset_(self):
        self.audio = None
        self.acArr = []
        self.nrgArr = []
        self.sigPwr = 0
        self.noisePwr = 0
        self.sigCnt = 0
        self.noiseCnt = 0
        self.conf = 0
        self.snr = 0

    def _concatenateChannels_(self):
        self.audio = np.array(self.audio)
        if len(self.audio.shape) == 0:
            raise ValueError("audio stream empty")
        elif len(self.audio.shape) == 1:
            pass
        elif self.audio.shape[1] > 1:
            self.audio = np.reshape(self.audio, self.audio.shape[0]*self.audio.shape[1], order='F')

    def _autoCorr_(self, frame, mode="half"):
        result = np.correlate(frame, frame, mode='full')
        if mode == "half":
            return result[int(result.size/2):]
        elif mode == "centered":
            return np.concatenate((result[:int(result.size/2)],result[int(result.size/2):]))
        else:
            return result

    def _computeAcNrg_(self):
        nFrames = int(np.ceil(len(self.audio)/self.hopSize))
        for i in range(nFrames):
            lowIdx = int(i * self.hopSize)
            uppIdx = int(min(lowIdx + self.frameSize, len(self.audio)))
            frame = self.audio[lowIdx:uppIdx]
            ac = abs(self._autoCorr_(frame, mode="half"))
            nrg = ac[0]
            ac = ac[0]/sum(ac) if sum(ac) > 0 else 0
            self.acArr.append(ac)
            self.nrgArr.append(nrg)
    
    def _normalise_(self):
        self.acArr /= max(self.acArr)
        self.nrgArr /= max(self.nrgArr)

    def _classifier_(self):
        for nrg, ac in zip(self.nrgArr, self.acArr):
            if nrg > self.nrgThreshold and ac < self.acThreshold:
                self.sigPwr += nrg**2
                self.sigCnt += 1
            else:
                self.noisePwr += nrg**2
                self.noiseCnt += 1

    def _compute_(self):
        if self.noiseCnt == 0:
            self.snr = np.inf
        elif self.sigCnt == 0:
            self.snr = 10 * np.log10(eps)
        else:
            if self.mode == "average":
                self.sigPwr /= self.sigCnt
                self.noisePwr /= self.noiseCnt
            self.snr = 10 * np.log10(self.sigPwr/self.noisePwr)

        self.conf = 1-abs(self.noiseCnt-self.sigCnt)/(self.sigCnt + self.noiseCnt)

    def visualise(self, **kwargs):
        fig, ax = plt.subplots(3, 1, **kwargs)
        ax[0].set_title("Audio")
        ax[0].plot(self.audio)
        
        ax[1].set_title("Energy Array")
        ax[1].plot(self.nrgArr)
        ax[1].hlines(self.nrgThreshold, xmin=0, xmax=len(self.nrgArr))

        ax[2].set_title("Correlation Array")
        ax[2].plot(self.acArr)
        ax[2].hlines(self.acThreshold, xmin=0, xmax=len(self.acArr))
        plt.show()
