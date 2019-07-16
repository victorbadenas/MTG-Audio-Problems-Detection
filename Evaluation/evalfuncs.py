import os
import utils as u
import matplotlib.pyplot as plt
from AudioFile import AudioFile
from essentia.standard import AudioLoader
import numpy as np
import simpleaudio as sa


def convert_to_bin_array(value: int, b: int):
    if value > 2**(b-1)-1:
        raise ValueError("Value too large")
    if value < -2**(b-1):
        raise ValueError("Value too small")
    bin_str = np.binary_repr(value, width=b)
    return [int(val) for val in bin_str]

def biteval(filename):
    audio = AudioFile(filename)
    cnt = [0]*audio.bitDepth
    audio = np.array([2**(audio.bitDepth-1)*v for v in audio])
    for v in audio[:1000]:
        bin = convert_to_bin_array(int(v), len(cnt))
        cnt = [i+j for i,j in zip(cnt,bin)]
    print(cnt)
    inp = input("Does this file have BIT DEPTH problems? [y/n]")
    return u.str2bool(inp)

def bandeval(filename):
    inp = input("Does this file have BANDWIDTH problems? [y/n]")
    return u.str2bool(inp)

def snreval(filename):
    inp = input("Does this file have SNR problems? [y/n]")
    return u.str2bool(inp)

def sateval(filename):
    inp = input("Does this file have SATURATION problems? [y/n]")
    return u.str2bool(inp)

def humeval(filename):
    inp = input("Does this file have HUM problems? [y/n]")
    return u.str2bool(inp)

def clickeval(filename):
    inp = input("Does this file have CLICK problems? [y/n]")
    return u.str2bool(inp)

def silenceeval(filename):
    inp = input("Does this file have SILENCE problems? [y/n]")
    return u.str2bool(inp)

def fseval(filename):
    return False
    audio, SR, channels, _, br, _ = AudioLoader(filename=filename)()
    if len(audio.shape) == 1:
        return False
    cov = np.cov(audio[:, 0], audio[:, 1])[0][1]
    stdL = np.std(audio[:, 0])
    stdR = np.std(audio[:, 1])
    if(stdR*stdL == 0):
        print("Pearson Correlation = 0")
    else:
        print("Pearson Correlation = ", str(np.clip(cov/(stdR*stdL), -1, 1)))
    inp = input("Does this file have FALSE STEREO problems? [y/n]")
    return u.str2bool(inp)

def oopeval(filename):
    return False
    audio, SR, channels, _, br, _ = AudioLoader(filename=filename)()
    if len(audio.shape) == 1:
        return False
    cov = np.cov(audio[:, 0], audio[:, 1])[0][1]
    stdL = np.std(audio[:, 0])
    stdR = np.std(audio[:, 1])
    if(stdR*stdL == 0):
        print("Pearson Correlation = 0")
    else:
        print("Pearson Correlation = ", str(np.clip(cov/(stdR*stdL), -1, 1)))
    inp = input("Does this file have OUT OF PHASE problems? [y/n]")
    return u.str2bool(inp)

def nbeval(filename):
    inp = input("Does this file have NOISE BURSTS problems? [y/n]")
    return u.str2bool(inp)

