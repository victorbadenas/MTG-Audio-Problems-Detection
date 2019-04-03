import essentia.standard as estd

def ess_hum_detector(x:list, Q0=.1, Q1=.55, detectionThreshold=1):
    """Computes the hum detection in x and computes a value over one of the path of the audio that has hum noise.
    
    Args:
        x: (list) input signal
        Q0: (float)
        Q1: (float) 
        detectionThreshold: (float)

    Returns:
        Part over one of the file whith hum noise
    """
    _, _, _, starts, ends = estd.HumDetector(Q0=.1, Q1=.55, detectionThreshold=detectionThreshold)(x)
    #print(starts,ends)

    dur = []
    for s,e in zip(starts,ends):
        dur.append(e-s)
    
    return sum(dur)/len(x)
