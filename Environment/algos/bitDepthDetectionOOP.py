import numpy as np
import matplotlib.pyplot as plt

class BitDepthDetector:
    def __init__(self, bitDepth, chunkLen, numberOfChunks):
        if bitDepth in [8, 16, 24, 32]:
            self.containerBitDepth = bitDepth
        else:
            raise ValueError("Only bit depths accepted are 8, 16, 24, 32")
        self.chunkLen = int(chunkLen)
        self.numberOfChunks = int(numberOfChunks)

    def __call__(self, audio):
        audio = self._concatenateChannels(audio)
        audio = self._convertAudio(audio)
        audio = np.clip(audio, -2**(self.containerBitDepth-1), 2**(self.containerBitDepth-1)-1)
        audioRandom = self._randomizeChunks(audio)
        usedBits = self. _computeUnusedBits(audioRandom)
        return self._decisor(usedBits)

    def _concatenateChannels(self, audio):
        audio = np.array(audio).transpose()
        if len(audio.shape) == 0:
            raise ValueError("audio stream empty")
        elif len(audio.shape) == 1:
            return audio
        elif audio.shape[1] > 1:
            monoAudio = []
            for channel in audio:
                monoAudio = [*monoAudio, *channel]
            return np.array(monoAudio)

    def _convertAudio(self, audio):
        audio = (2**(self.containerBitDepth-1)) * audio.astype('float64')
        if self.containerBitDepth == 8:
            audio = audio.astype('int8')
        elif self.containerBitDepth == 16:
            audio = audio.astype('int16')
        elif self.containerBitDepth == 24:
            audio = audio.astype('int32')
        elif self.containerBitDepth == 32:
            audio = audio.astype('int32')
        return audio

    def _randomizeChunks(self, audio):
        random_audio_splices = []
        for _ in range(self.numberOfChunks):
            start_idx = np.random.randint(0, len(audio-self.chunkLen-1))
            random_audio_splices = [*random_audio_splices, *audio[int(start_idx):int(start_idx+self.chunkLen)]]
        return random_audio_splices

    def _toBinArray(self, value):
        return [int(value) for value in np.binary_repr(value, width=self.containerBitDepth)]

    def _computeUnusedBits(self, audioRandom):
        # initialise logic array to b positions of 0
        usedBits = [0]*self.containerBitDepth

        for sample in audioRandom:
            bin_arr = self._toBinArray(sample)  # compute the binary number for each sample

            # if any bit is used once, it is converted to 1, only bits that remain unused
            # through all the samples are kept as 0
            usedBits = [a or b for a, b in zip(usedBits, bin_arr)]

        return usedBits

    def _decisor(self, usedBits):
        for i, el in enumerate(reversed(usedBits)):
            if el != 0:
                # the smallest position with a 0 will determine the number of unused bits
                return len(usedBits)-i, (len(usedBits)-i) < self.containerBitDepth
        return 0, True
