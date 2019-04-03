import sys
import os
import json
import argparse
from satDetection import *
import essentia.standard as estd


def main(audiopath, jsonpath=""):
    """Calls the audio_problems_detection algorithms and stores the result in a json file

    Args:
        audiopath: string containing the relative path for the audio file
        jsonpath: string containing the relative path for the json file

    """

    if not os.path.exists(audiopath): raise ValueError("Audio File does not exist")
    print("Essentia Modules installed:")
    print(dir(estd))

    audio, sr, channels, _, br, codec = estd.AudioLoader(filename = audiopath)()
    print("filename params: \n Sample Rate:{0}Hz \n Number of channels:{1} \n Bit Rate:{2} \n Codec:{3}".format(sr, channels, br, codec))

    starts, ends, count = ess_saturation_detector(audio)
    print("number of frames with clips:", count)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calls the audio_problems_detection algorithms and stores the result in a json file")
	parser.add_argument("audiopath", help="relative path to the audiofile")
	args = parser.parse_args()
	main(args.audiopath)