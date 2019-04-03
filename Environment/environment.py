import sys
import os
import json
import argparse
from satDetection import *
from humDetection import *
from clickDetection import *
from startstopDetection import *
import numpy as np
import essentia.standard as estd
import matplotlib.pyplot as plt


def main(audiopath, jsonpath=""):
    """Calls the audio_problems_detection algorithms and stores the result in a json file

    Args:
        audiopath: string containing the relative path for the audio file
        jsonpath: string containing the relative path for the json file

    """

    if not os.path.exists(audiopath): raise ValueError("Audio File does not exist")
    #print("Essentia Modules installed:")
    #print(dir(estd))

    audio, sr, channels, _, br, codec = estd.AudioLoader(filename = audiopath)()

    if channels > 1: audio = np.sum(audio, axis=1)/channels

    filename = os.path.basename(audiopath)

    print("{0} data: \n \tfilename params: \n \tSample Rate:{1}Hz \n \tNumber of channels:{2} \n \tBit Rate:{3} \n \tCodec:{4}".format(filename,sr, channels, br, codec))

    satstarts, satends, satperc = ess_saturation_detector(audio)
    humperc                     = ess_hum_detector(audio)
    clkstarts, clkends, clkperc = ess_click_detector(audio)
    silperc                     = ess_startstopdetector(audio)

    print("Saturation: \n \tStarts length: {0} \n \tEnds length: {1} \n \tPercentage of clipped frames: {2}%".format(len(satstarts), len(satends), satperc))
    print("Hum: \n \tPercentage of the file with Hum: {}%".format(humperc))
    print("Clicks: \n \tStarts length: {0} \n \tEnds length: {1} \n \tPercentage of clipped frames: {2}%".format(len(clkstarts), len(clkends), clkperc))
    print("Silence: \n \tPercentage of the file that is silence: {}%".format(silperc))

    json_dict = {
        "Saturation" : {
            "Start indexes" : satstarts,
            "End indexes" : satends,
            "Percentage" : satperc,
        },
        "Hum" : {
            "Percentage" : humperc,
        },
        "Clicks" : {
            "Start indexes" : clkstarts,
            "End indexes" : clkends,
            "Percentage" : clkperc,
        },
        "Silence" : {
            "Percentage" : silperc,
        },
        
    }
    print(json_dict)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calls the audio_problems_detection algorithms and stores the result in a json file")
	parser.add_argument("audiopath", help="relative path to the audiofile")
	args = parser.parse_args()
	main(args.audiopath)