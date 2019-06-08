import os
import json
import argparse
import numpy as np
from essentia.standard import AudioLoader
from algos.satDetection import essSaturationDetector
from algos.noiseDetection import essHumDetector, essNoiseburstDetector
from algos.clickDetection import essClickDetector
from algos.startstopDetection import essStartstopDetector
from algos.essPhaseDetection import essFalsestereoDetector, outofPhaseDetector
from algos.LowSnrOOP import LowSnrDetector
from algos.bitDepthDetectionOOP import BitDepthDetector
from algos.bwDetectionOOP import BwDetection


def single_json_compute(audioPath, jsonFolder, printFlag=False):
    """Calls the audio_problems_detection algorithms and stores the result in a json file

    Args:
        audioPath: string containing the relative path for the audio file
        jsonFolder: string containing the relative path for the json folder
        printFlag: (boolean) True if a preview of the josn file is desired, False otherwise (default = False)

    Returns:
        json_dict: (dict) dictionary with the audio's features
    """
    # print(audioPath)
    if not os.path.exists(audioPath):
        raise ValueError("Audio File does not exist")
    if not os.path.exists(jsonFolder):
        print(jsonFolder + " does not exist, defaulting to audiofolder: " + os.path.dirname(args.audioPath))
        jsonFolder = os.path.dirname(args.audioPath)
    
    # print("Essentia Modules installed:")
    # print(dir(estd))

    audio, sr, channels, _, br, _ = AudioLoader(filename=audioPath)()

    monoAudio = np.sum(audio, axis=1)/channels

    frameSize = int(1024)
    if len(monoAudio)/frameSize < 10:
        frameSize = int(2 ** np.ceil(np.log2(len(monoAudio)/10)))

    hopSize = int(frameSize/2)
    bitDepthContainer = int(br / sr / channels)

    filename = os.path.splitext(os.path.basename(audioPath))[0]

    snr = LowSnrDetector(frameSize=frameSize, hopSize=hopSize)
    bit = BitDepthDetector(bitDepth=bitDepthContainer, chunkLen=100, numberOfChunks=100)
    bandWidth = BwDetection()
    
    satStarts, satEnds, satPercentage, satBool = essSaturationDetector(monoAudio, frameSize=frameSize, hopSize=hopSize)
    humPercentage, humBool = essHumDetector(monoAudio, sr=sr)
    clkStarts, clkEnds, clkPercentage, clkBool = essClickDetector(monoAudio, frameSize=frameSize, hopSize=hopSize)
    nbIndexes, nbPercentage, nbBool = essNoiseburstDetector(monoAudio, frameSize=frameSize, hopSize=hopSize)
    if len(monoAudio) > 1465:
        silPercentage, silBool = essStartstopDetector(monoAudio, frameSize=frameSize, hopSize=hopSize)
    else:
        silPercentage, silBool = "Audio file too short", False
    fsPercentage, fsBool = essFalsestereoDetector(audio, frameSize=frameSize, hopSize=hopSize)
    oopPercentage, oopBool = outofPhaseDetector(audio, frameSize=frameSize, hopSize=hopSize)

    snr, snrBool = snr(audio)
    extrBitDepth, bitDepthBool = bit(audio)
    bwCutFrequency, bwConfidence, bwBool = bandWidth(audio, sr)

    info = {
        "Saturation": {"Start indexes": len(satStarts), "End indexes": len(satEnds),
                       "Percentage": satPercentage, "Bool": satBool},
        "Hum": {"Percentage": humPercentage, "Bool": humBool},
        "Clicks": {"Start indexes": len(clkStarts), "End indexes": len(clkEnds),
                   "Percentage": clkPercentage, "Bool": clkBool},
        "Silence": {"Percentage": silPercentage, "Bool": silBool},
        "FalseStereo": {"Percentage": fsPercentage, "Bool": fsBool},
        "OutofPhase": {"Percentage": oopPercentage, "Bool": oopBool},
        "NoiseBursts": {"Indexes": len(nbIndexes), "Percentage": nbPercentage, "Bool": nbBool},
        # "BitDepth": { "BitDepth": bitDepthBool, "extracted": extrBitDepth},
        # "Bandwidth": { "Bandwidth": bwBool, "cutfrequency": bwCutFrequency, "confidence": bwConfidence},
        # "lowSNR": { "lowSNR": snrBool, "SNR": snr}
        # "Saturation": {"Bool": satBool},
        # "Hum": {"Bool": humBool},
        # "Clicks": {"Bool": clkBool},
        # "Silence": {"Bool": silBool},
        # "FalseStereo": {"Bool": fsBool},
        # "OutofPhase": {"Bool": oopBool},
        # "NoiseBursts": {"Bool": silBool},
        "BitDepth": {"Bool": bitDepthBool},
        "Bandwidth": {"Bool": bwBool},
        "lowSNR": {"Bool": snrBool}
    }

    if printFlag:
        print("{0} data: \n \tfilename params: \n \tSample Rate:{1}Hz \n \tNumber of channels:{2} \
              \n \tBit Rate:{3}".format(filename, sr, channels, br))
        print("\n \tLength of the audio file: {0} \n \tFrame Size: {1} \n \tHop Size: {2}".format(
            len(monoAudio), frameSize, hopSize))

        for problem in info:
            string = "{}: \n".format(problem)
            for feature in info[problem]:
                string = "{}\t{}: {} \n".format(string, feature, info[problem][feature])
            print(string)

    jsonpath = os.path.join(jsonFolder, filename + ".json")
    with open(jsonpath, "w") as jsonfile:

        json_dict = info.copy()
        for problem in json_dict:
            if isinstance(json_dict[problem], dict):
                for feature in json_dict[problem]:
                    if feature == "Bool":
                        json_dict[problem]["Bool"] = str(json_dict[problem]["Bool"])

        json.dump(json_dict, jsonfile)
    
    return info


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Calls the audio_problems_detection algorithms \
                                                 and stores the result in a json file")
    parser.add_argument("audioPath", help="relative path to the audiofile")
    parser.add_argument("--jsonfolder", help="string containing the relative path for the json file",
                        default="", required=False)
    parser.add_argument("--printFlag", help="boolean, True if it is desired to print the information, False otherwise",
                        default=False, required=False)
    args = parser.parse_args()
    if args.jsonfolder == "":
        jsonfolder = os.path.dirname(args.audioPath)
        print("json folder is:", jsonfolder)
        single_json_compute(args.audioPath, jsonfolder, args.printFlag)
    else:
        single_json_compute(args.audioPath, args.jsonfolder, args.printFlag)