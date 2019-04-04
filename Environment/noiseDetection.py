import essentia.standard as estd

def ess_hum_detector(x:list, Q0=.1, Q1=.55, detectionThreshold=1):
    """Computes the hum detection in x and computes a value over one of the path of the audio that has hum noise.
    
    Args:
        x: (list) input signal
        Q0: (float)
        Q1: (float) 
        detectionThreshold: (float)

    Returns:
        Percentage of the file whith hum noise
    """
    _, _, _, starts, ends = estd.HumDetector(Q0=.1, Q1=.55, detectionThreshold=detectionThreshold)(x)
    #print(starts,ends)

    dur = []
    for s,e in zip(starts,ends):
        dur.append(e-s)
    
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

    return idxs, round(100*count/(i+1),2)