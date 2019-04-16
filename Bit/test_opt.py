import essentia.standard as estd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from essentia import array as esarr

def Bit_Detection(fpath:str):

    if os.path.splitext(fpath)[1] != ".wav":
        raise ValueError("file must be wav")

    audio, SR, channels, _, br, _ = estd.AudioLoader(filename=fpath)()

    b = int(br / SR / channels) #number of bits used to code the fpath signal

    if channels >= 1: audio = audio[:,0]
    audio = (2**b) * ((0.5*audio)+0.5)

    possible_b_array = []
    b_tmp = b - 8

    tolerance = 8

    while b_tmp>=8:
        possible_b_array.append(b_tmp)
        b_tmp -= 8

    chunk_len = 100
    number_of_chunks = 100
    positions = np.random.randint(0, len(audio)-chunk_len-1, size = number_of_chunks)

    audio_to_analyse = []
    for idx in positions:
        audio_to_analyse = [*audio_to_analyse, *audio[int(idx):int(idx+chunk_len)]]

    audio_to_analyse = [int(val) for val in audio_to_analyse]

    conf_arr = []
    for possible_b in possible_b_array:
        wrong = 0
        hop = 2 ** (b-possible_b)
        #tolerance = 8 - b/possible_b
        for val in audio_to_analyse:
            #if possible_b == 16: print(val % hop)
            #if ((val % hop) > tolerance) and ((val % hop) < (hop - tolerance)):
            #    wrong += 1
            if val%hop == 0 : wrong += 1
            
        conf = 1-wrong/len(audio_to_analyse)
        conf_arr.append(conf)
        print("b:{0}\tprob:{1}".format(possible_b, conf))
    
    print(possible_b_array, conf_arr)
    #(val % hop)<tol
    #(val % hop)>hop-tol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate correlation for all the sounds in s folder")
    parser.add_argument("directory", help="Directory of the files")
    args = parser.parse_args()
    Bit_Detection(args.directory)

    