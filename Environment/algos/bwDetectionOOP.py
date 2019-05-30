import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as estd
eps = np.finfo("float").eps


class BwDetection:
    def __init__(self, frameSize=256, hopSize=128, oversample=1, confTh=0.8):
        if not self.__isPower2(oversample):
            raise ValueError("oversample factor can only be 1, 2 or 4")
        self.frameSize = int(frameSize) * int(oversample)
        self.hopSize = int(hopSize)
        self.confTh = max(min(confTh, 1),0)

    def __reset__(self):
        self.avgFrames = None
        self.hist = None
        self.mostLikelyBin = None

    def __call__(self, audio, SR):
        self.__reset__()

        if audio.shape[1] != 1:
            audio = (audio[:, 0] + audio[:, 1]) / 2


        fcIndexArr = []
        self.hist = np.zeros(int(self.frameSize / 2 + 1))
        fft = estd.FFT(size=self.frameSize)  # declare FFT function
        window = estd.Windowing(size=self.frameSize, type="hann")  # declare windowing function
        self.avgFrames = np.zeros(int(self.frameSize / 2) + 1)

        maxNrg = max([sum(abs(fft(window(frame))) ** 2) for frame in
                       estd.FrameGenerator(audio, frameSize=self.frameSize, hopSize=self.hopSize, startFromZero=True)])

        for i, frame in enumerate(estd.FrameGenerator(audio, frameSize=self.frameSize, hopSize=self.hopSize, startFromZero=True)):

            frame = window(frame)  # apply window to the frame
            frameFft = abs(fft(frame))
            nrg = sum(frameFft ** 2)

            if nrg >= 0.1*maxNrg:
                for j in reversed(range(len(frameFft))):
                    if sum(frameFft[j:]/j) >= 1e-5:
                        fcIndexArr.append(j)
                        self.hist[j] += nrg
                        break
                self.avgFrames = self.avgFrames + frameFft

        if len(fcIndexArr) == 0:
            fcIndexArr.append(int(self.frameSize / 2) + 1)
            self.hist[int(self.frameSize / 2)] += 1

        self.avgFrames /= (i+1)
        self.mostLikelyBin, conf, binary = self.__computeMeanFc(fcIndexArr, np.arange(int(self.frameSize/2)+2), hist=self.hist)

        return self.mostLikelyBin*SR/self.frameSize, conf, binary
        # print("f={:0=2f}, conf={:0=2f}, problem={}".format(
        #     self.mostLikelyBin*SR / self.frameSize, conf, str(binary)))

    def __computeMeanFc(self, fcIndexArr, xticks, hist=np.array([])):
        """computes the most possible fc for that audio file

        Args:
            fcIndexArr: iterable with the predicted fc's per frame
            xticks: iterable containing the bin2f array
            SR: (float) Sample Rate

        Returns:
            most_likely_f: (float) frequency corresponding to the highest peak in the histogram
            conf: (float) confidence value between 0 and 1
            (bool): True if the file is predicted to have the issue, False otherwise
        """
        if len(hist) == 0:
            hist = self.__computeHistogram(fcIndexArr, xticks)  # computation of the histogram

        hist[hist < .1*max(hist)] = 0
        mostLikelyBin = np.argmax(hist)  # bin value of the highest peak of the histogram
        # mostLikelyBin = get_last_true_index(hist != 0)
        # the confidence value changes depending on if mostLikelyBin falls under the 85% lower spectrogram or not
        if mostLikelyBin <= .85*len(hist):
            # if it falls under the 85% lower, the confidence is computed by a weighted sum of the values of the histogram,
            # the highest peak having the highest importance and decreasing as the indexes go further.
            """
            #creation of the confidence scale
            conf_scale = abs(mostLikelyBin - np.arange(len(hist))); conf_scale = max(conf_scale) - conf_scale ; conf_scale = conf_scale / max(conf_scale)
            conf = sum(hist * conf_scale) / sum(hist) #computation of the confidence sum, normalised by the histogram length
            
            conf_scale = abs(mostLikelyBin - np.arange(len(hist))); conf_scale = conf_scale / max(conf_scale); conf_scale = conf_scale**(1/5)
            plt.plot(conf_scale); plt.show()
            conf = 1 - sum(hist * conf_scale) / sum(hist)"""
            conf = sum(hist[mostLikelyBin-1:mostLikelyBin+1]) / sum(hist[:mostLikelyBin+1])
            # conf = sum(hist[int(mostLikelyBin-int(0.016*len(hist))):int(mostLikelyBin+int(0.016*len(hist)))])/sum(hist)
            # return the analog frequency corresponding to the bin, confidence value, and True if the confidence value is higher than 0.77
            return mostLikelyBin, conf, conf > self.confTh
        else:
            #if it falls over the 85% mark, the confidence is computated by summing the square of the 3 samples of the histogram closer to the max
            #and compare it to the sum of all the values appended to the histogram.
            conf = sum(hist[int(.85*len(hist)):]) / sum(hist)
            # print(conf)
            return mostLikelyBin, conf, False

    def __computeHistogram(self, idxArr, xticks, mask=[]):
        """Computes the histogram of an array of ints

        Args:
            idxArr: (iterable) with int values
            xticks: (iterable) with x values

        KWargs:
            mask: (iterable) int binary array

        Returns:
            hist: histogram
        """

        idxArr = [int(item) for item in idxArr]
        if len(mask) == 0:
            hist = np.zeros(len(xticks))
            for idx in idxArr:
                hist[idx] += 1
        else:
            hist = np.zeros(len(xticks))
            if len(mask) != len(idxArr):
                raise ValueError("Inconsistent mask size")
            for i, boolean in zip(idxArr, mask):
                if boolean:
                    hist[i] += 1
        return hist

    def __isPower2(self, num):
        return num != 0 and ((num & (num - 1)) == 0)

    def plot(self, **kwargs):

        fig, ax = plt.subplots(2, 1, **kwargs)
        ax[0].plot(20 * np.log10(self.avgFrames + eps))
        ax[0].axvline(x=self.mostLikelyBin, color='r')
        ax[0].set_ylim(bottom=-120)
        ax[1].stem(self.hist)
        plt.show()
