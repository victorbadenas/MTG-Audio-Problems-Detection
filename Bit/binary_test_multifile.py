import argparse
import os
import essentia.standard as estd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def convert_to_bin_array(value:int, b:int):
    if value > 2**(b-1)-1: raise ValueError("Value too large")
    if value < -2**(b-1): raise ValueError("Value too small") 
    
    bin_str = np.binary_repr(value, width = b)
    return [int(val) for val in bin_str]

def Bit_Detection_Binary(audio:list, b:int, chunk_len = 100, number_of_chunks = 100):

    if b not in [8,16,24,32]:
        raise ValueError("Only bit depths accepted are 8, 16, 24, 32")

    #set audio to be ints from -2**(b-1) to 2**(b-1)-1
    #and change type to int32 (32 bit is the highest coding depth allowed)
    audio = (2**(b-1)) * audio.astype('float64')
    audio = np.clip(audio,-2**(b-1),2**(b-1)-1)
    
    if b == 8: 
        audio = audio.astype('int8')
    elif b == 16: 
        audio = audio.astype('int16')
    elif b == 24: 
        audio = audio.astype('int32')
    elif b == 32: 
        audio = audio.astype('int32')
    else: 
        audio = audio.astype('int64')
    
    #get number_of_chunks random splices of data of chunk_len samples each one
    positions = np.random.randint(0, len(audio)-chunk_len-1, size = number_of_chunks)

    audio_to_analyse = []
    for idx in positions:
        audio_to_analyse = [*audio_to_analyse, *audio[int(idx):int(idx+chunk_len)]]

    result = [0]*b
    for sample in audio_to_analyse:
        bin_arr = convert_to_bin_array(sample,b)
        result = [ a or b for a,b in zip(result,bin_arr)]
    #print(result)
    for i,el in enumerate(reversed(result)):
        if el != 0:
            return len(result)-i

def Bit_Detection_multifile(folder:str):
    if not os.path.exists(folder):
        raise ValueError("{} does not exist".format(folder))
    
    df = pd.DataFrame()

    for file in os.listdir(folder):

        if os.path.splitext(file)[1] != ".wav":
            print("{} skipped because it was not a wav file".format(file))
            continue

        fpath = os.path.join(folder,file)
        audio, SR, channels, _, br, _ = estd.AudioLoader(filename=fpath)()

        if channels >= 1: audio = audio[:,0]
        
        b = int(br / SR / channels) #number of bits used to code the fpath signal

        extracted_b = Bit_Detection_Binary(audio, b)
        #correct_b = min(b,int(file.split('b')[0]))
        df_temp = pd.DataFrame({
            "Filename" : [file],
            "Container" : [b],
            #"Correct" : [correct_b],
            "Extracted" : [extracted_b],
            "Problem in file" : [extracted_b<b],
            #"Extracted_correctly" : [correct_b==extracted_b]
        })
        df = df.append(df_temp)
        #print("{}:\tcontainer_bits:{}\tcorrect_bits:{}\textracted_bits:{}\tcorrect:{}".format(file, b, correct_b, extracted_b, correct_b==extracted_b))
    
    df = df.set_index("Filename")
    print(df)
    with open("results.tsv", "w") as tsv:
        df.to_csv(tsv,sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate correlation for all the sounds in s folder")
    parser.add_argument("folder", help="Directory of the files")
    args = parser.parse_args()
    Bit_Detection_multifile(args.folder)