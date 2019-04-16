import argparse
import os
import essentia.standard as estd
import matplotlib.pyplot as plt
import numpy as np

def convert_to_bin_array(value:int, b:int):
    if value > 2**(b-1)-1: raise ValueError("Value too large")
    if value < -2**(b-1): raise ValueError("Value too small") 
    
    bin_str = np.binary_repr(value, width = b)
    return [int(val) for val in bin_str]

def Bit_Detection_Binary(fpath:str):

    if os.path.splitext(fpath)[1] != ".wav":
        raise ValueError("file must be wav")
    
    if not os.path.exists(fpath): 
        raise ValueError("file does not exist")

    audio, SR, channels, _, br, _ = estd.AudioLoader(filename=fpath)()

    b = int(br / SR / channels) #number of bits used to code the fpath signal

    if b > 32: 
        raise ValueError("Maximum bit depth allowed is 32bit")

    if channels >= 1: audio = audio[:,0] #if audio is stereo, only get the left channeñ
    
    #set audio to be ints from -2**(b-1) to 2**(b-1)-1
    #and change type to int32 (32 bit is the highest coding depth allowed)
    audio = (2**(b-1)) * audio.astype('float64')
    if b == 8: audio = audio.astype('int8')
    elif b == 16: audio = audio.astype('int16')
    elif b == 24: audio = audio.astype('int32')
    elif b == 32: audio = audio.astype('int32')
    else: audio = audio.astype('int64')
    #get 100 random splices of data of 100 samples each one
    chunk_len = 100
    number_of_chunks = 100
    positions = np.random.randint(0, len(audio)-chunk_len-1, size = number_of_chunks)

    audio_to_analyse = []
    for idx in positions:
        audio_to_analyse = [*audio_to_analyse, *audio[int(idx):int(idx+chunk_len)]]

    result = [0]*b
    for sample in audio_to_analyse:
        bin_arr = convert_to_bin_array(sample,b)
        result = [ a+b for a,b in zip(result,bin_arr)]
    print(result)
    print(sum([1 for el in result if el != 0]))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate correlation for all the sounds in s folder")
    parser.add_argument("directory", help="Directory of the files")
    args = parser.parse_args()
    Bit_Detection_Binary(args.directory)