import sys
import os

scriptFolder = os.path.dirname(os.path.realpath(__file__))
algosFolder = os.path.join(scriptFolder, '../Environment/algos')
sys.path.append(algosFolder)

from satDetection import essSaturationDetector
from bwDetectionOOP import BwDetection
from LowSnrOOP import LowSnrDetector
from clickDetection import essClickDetector
from startstopDetection import essStartstopDetector
from noiseDetection import essHumDetector, essNoiseburstDetector
import pandas as pd
import utils as u
import logging
import numpy as np
import essentia.standard as std

Fbeta = 0.5

class gridSearch():
    def __init__(self, wavDatasetPath, groundTruthPath):
        if not os.path.exists(wavDatasetPath):
            raise ValueError("wavDatasetPath does not exiosts, please check the esistance of the path")
        if not os.path.exists(groundTruthPath):
            raise ValueError("groundTruthPath does not exiosts, please check the esistance of the path")
        self.wavDatasetPath = wavDatasetPath
        self.groundTruthPath = groundTruthPath
        self.files = [os.path.join(wavDatasetPath, file) for file in os.listdir(
            wavDatasetPath) if os.path.splitext(file)[-1] == ".wav"]
        self.loadGroundTruth(groundTruthPath)

    def __str__(self):
        return str(self.__dict__)

    def loadGroundTruth(self, groundTruthPath):
        self.groundTruth = pd.read_csv(groundTruthPath, index_col="Filename")

    def __call__(self):
        self.saturation()
        self.bandwidth()
        self.lowsnr()
        self.hum()
        self.clicks()
        self.silence()
        self.noisebursts()

    def saturation(self):
        satEnergyThreshold = [-30, -20, -10, -7, -5, -3, -1, -0.01]
        satDifferentialThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        satMinimumDuration = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in satEnergyThreshold:
            print("energy Threshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end = '\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, _, _, ret = essSaturationDetector(audio, energyThreshold=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Saturation")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/satEnergyThreshold.png", precision=precisionArr, recall=recallArr, Fscore=FscoreArr, x_values=satEnergyThreshold)

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in satDifferentialThreshold:
            print("energy Threshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, _, _, ret = essSaturationDetector(audio, differentialThreshold=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Saturation")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/satDifferentialThreshold.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=satDifferentialThreshold)
        
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in satMinimumDuration:
            print("energy Threshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, _, _, ret = essSaturationDetector(audio, minimumDuration=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Saturation")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/satMinimumDuration.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=satMinimumDuration)

    def bandwidth(self):
        BWsumThreshold = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]

        bandWidth = BwDetection()

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in BWsumThreshold:
            print("sumThreshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, _, ret = bandWidth(audio, sr, sumThreshold=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Bandwidth")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/BWsumThreshold.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=BWsumThreshold)

        BWConfTh = [0.6, 0.7, 0.8, 0.9]
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in BWConfTh:
            print("ConfTh: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                bandWidth = BwDetection(confTh=value)
                _, _, ret = bandWidth(audio, sr)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Bandwidth")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/BWConfTh.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=BWConfTh)

    def lowsnr(self):
        snrnrgThresholdArr = [0.1, 0.3, 0.5, 0.7, 0.9]
        snracThresholdArr = [0.1, 0.3, 0.5, 0.7, 0.9]
        snrThresholdArr = [-3, -1, 1, 3, 5, 7, 9]

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in snrnrgThresholdArr:
            print("sumThreshold: {} being evaluated".format(value))
            valueResults = []
            lsd = LowSnrDetector(nrgThreshold=value)
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, ret = lsd(audio)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "lowSNR")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/snrnrgThreshold.png", precision=precisionArr, recall=recallArr, Fscore=FscoreArr, x_values=snrnrgThresholdArr)

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in snracThresholdArr:
            print("sumThreshold: {} being evaluated".format(value))
            valueResults = []
            lsd = LowSnrDetector(acThreshold=value)
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename,
                                                              i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(
                    filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, ret = lsd(audio)
                valueResults.append(
                    (filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(
                valueResults, "lowSNR")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision *
                             recall / (Fbeta**2 * precision + recall))
        u.plot("./results/snracThreshold.png", precision=precisionArr,
               recall=recallArr, Fscore=FscoreArr, x_values=snracThresholdArr)

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in snrThresholdArr:
            print("sumThreshold: {} being evaluated".format(value))
            valueResults = []
            lsd = LowSnrDetector(snrThreshold=value)
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename,
                                                              i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(
                    filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, ret = lsd(audio)
                valueResults.append(
                    (filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(
                valueResults, "lowSNR")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision *
                             recall / (Fbeta**2 * precision + recall))
        u.plot("./results/snrThreshold.png", precision=precisionArr,
               recall=recallArr, Fscore=FscoreArr, x_values=snrThresholdArr)
               
    def hum(self):
        timeWindow = [0.1, 0.3, 0.5, 1, 3, 5]
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in timeWindow:
            print("sumThreshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, ret = essHumDetector(audio, timeWindow=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Hum")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/HumTimeWindow.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=timeWindow)

        minimumDuration = [0.01, 0.07, 0.1, 0.3, 0.5, 1, 3, 5]
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in minimumDuration:
            print("sumThreshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, ret = essHumDetector(audio, minimumDuration=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Hum")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/HumminimumDuration.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=minimumDuration)

        timeContinuity = [0.1, 0.3, 0.5, 1, 3, 5]
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in timeContinuity:
            print("sumThreshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, ret = essHumDetector(audio, timeContinuity=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Hum")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/HumtimeContinuity.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=timeContinuity)

        numberHarmonics = [i for i in range(6)]
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in numberHarmonics:
            print("sumThreshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, ret = essHumDetector(audio, numberHarmonics=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Hum")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/HumnumberHarmonics.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=numberHarmonics)

    def clicks(self):
        order = [int(2*i) for i in range(1,20)]
        detectionThreshold = [0, 5, 10, 15, 20, 25, 30, 35]
        powerEstimationThreshold = [int(2*i) for i in range(1, 8)]
        silenceThreshold = [-1*int(10*i) for i in range(1, 8)][::-1]

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in order:
            print("order: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, _, _, ret = essClickDetector(audio, order=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Clicks")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/clicksorder.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=order)

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in detectionThreshold:
            print("detectionThreshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, _, _, ret = essClickDetector(audio, detectionThreshold=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Clicks")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/clicksdetectionThreshold.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=detectionThreshold)

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in powerEstimationThreshold:
            print("powerEstimationThreshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, _, _, ret = essClickDetector(audio, powerEstimationThreshold=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Clicks")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/clickspowerEstimationThreshold.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=powerEstimationThreshold)

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in silenceThreshold:
            print("silenceThreshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, _, _, ret = essClickDetector(audio, silenceThreshold=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Clicks")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/clickssilenceThreshold.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=silenceThreshold)

    def silence(self):
        threshold = [-1*int(10*i) for i in range(1, 11)][::-1]
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in threshold:
            print("threshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, ret = essStartstopDetector(audio, threshold=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Clicks")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/silencethreshold.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=threshold)

        frameSize = [int(2**i) for i in range(5,10)]
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in frameSize:
            print("frameSize: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, ret = essStartstopDetector(audio, frameSize=value, hopSize=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "Clicks")
            precisionArr.append(precision)
            recallArr.append(recall)
            FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/silenceframeSize.png", precision=precisionArr, recall=recallArr, Fscore=FscoreArr, x_values=frameSize)

    def noisebursts(self): 
        threshold = [2*i for i in range(-5,6)]
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in threshold:
            print("threshold: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, _, ret = essNoiseburstDetector(audio, threshold=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "NoiseBursts")
            precisionArr.append(precision)
            recallArr.append(recall)
            if (precision + recall) == 0.0:
                FscoreArr.append(0.0)
            else:
                FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/noisethreshold.png", precision=precisionArr, recall=recallArr, Fscore=FscoreArr, x_values=threshold)

        alpha = [i/10 for i in range(1,10)]
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in alpha:
            print("alpha: {} being evaluated".format(value))
            valueResults = []
            for i, filename in enumerate(self.files):
                print("Executing file {} number {}/{}".format(filename, i+1, len(self.files)), end='\r')
                audio, sr, channels, _, _, _ = std.AudioLoader(filename=filename)()
                audio = np.sum(audio, axis=1)/channels
                _, _, ret = essNoiseburstDetector(audio, alpha=value)
                valueResults.append((filename.replace(self.wavDatasetPath, ""), ret))
            print('')
            valueResults = sorted(valueResults, key=lambda x: x[0])
            _, precision, recall = self.evaluateValue(valueResults, "NoiseBursts")
            precisionArr.append(precision)
            recallArr.append(recall)
            if (precision + recall) == 0.0:
                FscoreArr.append(0.0)
            else:
                FscoreArr.append((1 + Fbeta**2) * precision * recall / (Fbeta**2 * precision + recall))
        u.plot("./results/noiseealpha.png", precision=precisionArr,
               recall=recallArr, Fscore=FscoreArr, x_values=alpha)

    def evaluateValue(self, valueResults, label):
        gtlist = self.groundTruth[label].sort_index().tolist()
        boolResults = [i[1] for i in valueResults]
        ret = {"truePositive": 0, "falsePositive": 0, "trueNegative": 0, "falseNegative": 0}
        for val, gt in zip(boolResults, gtlist):
            if val and gt:
                ret["truePositive"]+=1
            elif val and not gt:
                ret["falsePositive"]+=1
            elif not val and gt:
                ret["falseNegative"]+=1
            else:
                ret["trueNegative"]+=1
        if ret["truePositive"] + ret["falsePositive"] != 0:
            precision = float(ret["truePositive"]) / float(ret["truePositive"] + ret["falsePositive"])
        else:
            precision = 1
        if ret["truePositive"] + ret["falseNegative"] != 0:
            recall = float(ret["truePositive"]) / float(ret["truePositive"] + ret["falseNegative"])
        else:
            recall = 1
        print(ret)
        print("precision:",precision)
        print("recall:",recall)
        return ret, precision, recall


if __name__ == "__main__":
    gs = gridSearch("../Dataset/test/", "../Evaluation/results/evalresults.csv")
    gs()
    # print(str(gs))
