import sys
import os
import pandas as pd
import utils as u
import logging
import numpy as np
import essentia.standard as std

scriptFolder = os.path.dirname(os.path.realpath(__file__))
algosFolder = os.path.join(scriptFolder, '../Environment/algos')
sys.path.append(algosFolder)

from satDetection import essSaturationDetector

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
        self.bitdepth()
        self.bandwidth()
        self.lowsnr()
        self.saturation()
        self.hum()
        self.clicks()
        self.silence()
        self.falsestereo()
        self.outofphase()
        self.noisebursts()

    def saturation(self):
        energyThreshold = [-30, -20, -10, -7, -5, -3, -1, -0.01]
        differentialThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        minimumDuration = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in energyThreshold:
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
        u.plot("./results/energyThreshold.png", precision=precisionArr, recall=recallArr, Fscore=FscoreArr, x_values=energyThreshold)

        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in differentialThreshold:
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
        u.plot("./results/differentialThreshold.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=differentialThreshold)
        
        precisionArr = []
        recallArr = []
        FscoreArr = []
        for value in minimumDuration:
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
        u.plot("./results/minimumDuration.png", precision=precisionArr,recall=recallArr, Fscore=FscoreArr, x_values=minimumDuration)

    def bitdepth(self):
        pass

    def bandwidth(self):
        pass

    def lowsnr(self):
        pass

    def hum(self):
        pass

    def clicks(self):
        pass

    def silence(self):
        pass

    def falsestereo(self):
        pass

    def outofphase(self):
        pass

    def noisebursts(self):
        pass

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
            precision = 0
        if ret["truePositive"] + ret["falseNegative"] != 0:
            recall = float(ret["truePositive"]) / float(ret["truePositive"] + ret["falseNegative"])
        else:
            recall = 0
        print(ret)
        print("precision:",precision)
        print("recall:",recall)
        return ret, precision, recall




if __name__ == "__main__":
    gs = gridSearch("../Dataset/test/", "../Evaluation/results/evalresults.csv")
    gs()
    # print(str(gs))
