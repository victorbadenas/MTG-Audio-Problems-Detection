import os
import pandas as pd
from single_file_json_compute import *
import json

def multi_file_compute(audiofolder, jsonfolder):
    """Calls the audio_problems_detection algorithms and stores the result in a json file

    Args:
        audiofolder: string containing the relative path for the folder containing the audio files
        jsonfolder: string containing the relative path for the folder containing the json files

    """

    for file in os.listdir(audiofolder):
        _, extension = os.path.splitext(file)
        if extension == ".wav":
            audiopath = os.path.join(audiofolder,file)
            single_json_compute(audiopath, jsonfolder, print_flag = True)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Calls the audio_problems_detection algorithms and stores the result in a json file")
    parser.add_argument("audiofolder", help="relative path to the audiofile")
    parser.add_argument("--jsonfolder", help="string containing the relative path for the json file", default="",required=False)
    args = parser.parse_args()

    if args.jsonfolder == "":
        print("json folder is:", args.audiofolder)
        multi_file_compute(args.audiofolder,args.audiofolder)
    else:
        multi_file_compute(args.audiofolder,args.jsonfolder)