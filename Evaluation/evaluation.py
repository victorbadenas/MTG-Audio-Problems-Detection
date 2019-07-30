import os
import numpy as np
from evalfuncs import *

def main(datasetPath):

    if not os.path.exists("results/"):
        os.mkdir("results/")

    outputfile = "results/evalresults.csv"

    alreadyRunFiles = set()
    if os.path.exists(outputfile):
        with open(outputfile, "r") as f:
            for line in f:
                fname = line.split(",")[0]
                if os.path.splitext(fname)[-1] == ".wav":
                    alreadyRunFiles.add(fname)
    else:
        with open(outputfile, "a") as f:
            line = "Filename,BitDepth,Bandwidth,lowSNR,Saturation,Hum,Clicks,Silence,FalseStereo,OutofPhase,NoiseBursts\n"
            f.write(line)

    for filename in os.listdir(datasetPath):
        if os.path.splitext(filename)[-1] != ".wav":
            continue
        if filename in alreadyRunFiles:
            print(filename + " has already been evaluated")
            continue
        fullfilename = os.path.join(datasetPath, filename)
        print("Now evaluating: ", filename)
        audio, SR, channels, _, br, _ = AudioLoader(filename=fullfilename)()
        if channels > 1:
            fig, ax = plt.subplots(channels + 1, 1)
            for i in range(channels):
                ax[i].plot(audio[:i])
            monoAudio = np.sum(audio, axis=1)/channels
            ax[-1].plot(abs(np.fft.fft(monoAudio)))
            plt.show(block=False)
        else:
            audio = audio[:,0]
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(audio)
            ax[1].plot(20*np.log10(abs(np.fft.fft(audio)[:int(len(audio)/2)])))
            plt.show(block=False)

        wave_obj = sa.WaveObject.from_wave_file(fullfilename)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        results = {
            "BitDepth": biteval(fullfilename) ,
            "Bandwidth": bandeval(fullfilename) ,
            "lowSNR": snreval(fullfilename) ,
            "Saturation": sateval(fullfilename) ,
            "Hum": humeval(fullfilename) ,
            "Clicks": clickeval(fullfilename) ,
            "Silence": silenceeval(fullfilename) ,
            "FalseStereo": fseval(fullfilename) ,
            "OutofPhase": oopeval(fullfilename) ,
            "NoiseBursts": nbeval(fullfilename)
        }
        plt.close()
        with open("results/evalresults.csv", "a") as f:
            line = filename + "," + \
                ",".join([str(v) for k, v in results.items()]) + "\n"
            f.write(line)

if __name__=="__main__":
    datasetPath = "../Dataset/test"
    main(datasetPath)


