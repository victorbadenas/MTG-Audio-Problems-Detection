import sys
import os
import json
import argparse
from satDetection import *
from noiseDetection import *
from clickDetection import *
from startstopDetection import *
from phaseDetection import *
import numpy as np
import essentia.standard as estd
import matplotlib.pyplot as plt


def single_json_compute(audiopath, jsonfolder, print_flag=False):
    """Calls the audio_problems_detection algorithms and stores the result in a json file

    Args:
        audiopath: string containing the relative path for the audio file
        jsonfolder: string containing the relative path for the json folder
        print_flag: (boolean) True if a preview of the josn file is desired, False otherwise (default = False)

    """

    if not os.path.exists(audiopath): raise ValueError("Audio File does not exist")
    if not os.path.exists(jsonfolder): 
        print(jsonfolder + " does not exist, defaulting to audiofolder: " + os.path.dirname(args.audiopath))
        jsonfolder = os.path.dirname(args.audiopath)
    
    #print("Essentia Modules installed:")
    #print(dir(estd))

    audio, sr, channels, _, br, codec = estd.AudioLoader(filename = audiopath)()

    monoaudio = np.sum(audio, axis=1)/channels

    filename = os.path.basename(audiopath)
    filename, _ = os.path.splitext(filename)
    
    sat_starts, sat_ends, sat_perc = ess_saturation_detector(monoaudio)
    hum_perc                       = ess_hum_detector(monoaudio)
    clk_starts, clk_ends, clk_perc = ess_click_detector(monoaudio)
    fs_bool, fs_perc               = ess_falsestereo_detector(audio)
    oop_bool, oop_perc             = ess_outofphase_detector(audio)
    nb_indexes, nb_perc            = ess_noiseburst_detector(monoaudio)
    sil_perc                       = ess_startstop_detector(monoaudio) if len(monoaudio) > 1465 else "Audio file too short"
    
    if print_flag:
        print("{0} data: \n \tfilename params: \n \tSample Rate:{1}Hz \n \tNumber of channels:{2} \n \tBit Rate:{3} \n \tCodec:{4}".format(filename,sr, channels, br, codec))
        print("Saturation: \n \tStarts length: {0} \n \tEnds length: {1} \n \tPercentage of clipped frames: {2}%".format(len(sat_starts), len(sat_ends), sat_perc))
        print("Hum: \n \tPercentage of the file with Hum: {}%".format(hum_perc))
        print("Clicks: \n \tStarts length: {0} \n \tEnds length: {1} \n \tPercentage of clipped frames: {2}%".format(len(clk_starts), len(clk_ends), clk_perc))
        print("Silence: \n \tPercentage of the file that is silence: {}%".format(sil_perc))
        print("FalseStereo: \n \tIs falseStereo?: {0} \n \tPercentage of frames with correlation==1: {1}%".format(fs_bool,fs_perc))
        print("OutofPhase: \n \tIs outofphase?: {0} \n \tPercentage of frames with correlation<-0.8: {1}%".format(oop_bool, oop_perc))
        print("NoiseBursts: \n \tIndexes length:{0} \n \tPercentage of problematic frames: {1}%".format(len(nb_indexes),nb_perc))
    
    json_dict = {
        "Saturation" : {
            "Start indexes" : len(sat_starts),
            "End indexes" : len(sat_ends),
            "Percentage" : sat_perc
        },
        "Hum" : {
            "Percentage" : hum_perc
        },
        "Clicks" : {
            "Start indexes" : len(clk_starts),
            "End indexes" : len(clk_ends),
            "Percentage" : clk_perc
        },
        "Silence" : {
            "Percentage" : sil_perc
        },
        "FalseStereo" : {
            "Bool" : fs_bool,
            "Percentage" : fs_perc
        },
        "OutofPhase" : {
            "Bool" : oop_bool,
            "Percentage" : oop_perc
        },
        "NoiseBursts" : {
            "Indexes" : len(nb_indexes),
            "tPercentage" : nb_perc
        }
    }

    jsonpath = os.path.join(jsonfolder,filename + ".json")
    with open(jsonpath, "w") as jsonfile:
        json.dump(json_dict, jsonfile)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Calls the audio_problems_detection algorithms and stores the result in a json file")
    parser.add_argument("audiopath", help="relative path to the audiofile")
    parser.add_argument("--jsonfolder", help="string containing the relative path for the json file", default="",required=False)
    parser.add_argument("--print_flag", help="boolean, True if it is desired to print the information, False otherwise", default=False,required=False)
    args = parser.parse_args()
    if args.jsonfolder == "":
        jsonfolder = os.path.dirname(args.audiopath)
        print("json folder is:", jsonfolder)
        single_json_compute(args.audiopath,jsonfolder,args.print_flag)
    else:
        single_json_compute(args.audiopath,args.jsonfolder,args.print_flag)
	