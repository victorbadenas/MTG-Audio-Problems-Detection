import essentia.standard as estd
import numpy as np

def ess_hum_detector(x: list, Q0=.1, Q1=.55, frame_size=1024, hop_size=512, detectionThreshold=1, sr=44100):
    """Computes the hum detection in x and computes a value over one of the path of the audio that has hum noise.
    
    Args:
        x: (list) input signal
        Q0: (float)
        Q1: (float) 
        detectionThreshold: (float)

    Returns:
        Percentage of the file whith hum noise
    """
    minLength = 10*sr
    #print(len(x), minLength)
    if len(x) < minLength:
        x = np.tile(x,int(np.ceil(minLength/len(x))))
    #print(len(x))
    frame_size = frame_size / sr
    hop_size = hop_size / sr
    minimumDuration=frame_size / 4
    timeWindow = frame_size
    #print(minimumDuration, frame_size)
    _, _, _, starts, ends = estd.HumDetector(Q0=Q0, Q1=Q1, frameSize=frame_size, hopSize=hop_size, detectionThreshold=detectionThreshold, sampleRate=sr, minimumDuration=minimumDuration, timeWindow=timeWindow)(x)
    
    """
    try:
        if frame_size + hop_size == '':
            _, _, _, starts, ends = estd.HumDetector(Q0=Q0, Q1=Q1, detectionThreshold=detectionThreshold, sampleRate=sr, minimumDuration=0.1*len(x))(x)
        else:
            frame_size = frame_size / sr
            hop_size = hop_size / sr
            _, _, _, starts, ends = estd.HumDetector(Q0=Q0, Q1=Q1, frameSize=frame_size, hopSize=hop_size, detectionThreshold=detectionThreshold, sampleRate=sr, minimumDuration=0.1*len(x))(x)
    except(RuntimeError):
        return "error"
    """
    #print(starts,ends)

    dur = []
    for s,e in zip(starts,ends):
        dur.append(e-s)
    
    #len_x = len(x)
    #del starts; del ends; del _; del x
    return round(100*sum(dur)/len(x),2)

def ess_noiseburst_detector(x:list, frame_size=1024, hop_size=512, detection_th = 0.005):
    """Computes the hum detection in x and computes a value over one of the path of the audio that has hum noise.
    
    Args:
        x: (list) input signal
        Q0: (float)
        Q1: (float) 
        detectionThreshold: (float)

    Returns:
        Part over one of the file whith hum noise
    """
    noiseBurstDetector_algo = estd.NoiseBurstDetector()

    idxs = []
    count = 0
    for i,frame in enumerate(estd.FrameGenerator(x, frameSize=frame_size, hopSize=hop_size, startFromZero=True)):
        corrupt_samples = noiseBurstDetector_algo(frame)
        corrupt_samples = hop_size * i + corrupt_samples

        if len(corrupt_samples) > int(0.05*frame_size): 
            count += 1
            for s in corrupt_samples:
                idxs.append(s)
    
    #del noiseBurstDetector_algo; del frame; del corrupt_samples; del x;
    return idxs, round(100*count/(i+1),2)
