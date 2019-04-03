import essentia.standard as estd

def ess_saturation_detector(x:list, frame_size=1024, hop_size=512, **kwargs):
    """Breaks x into frames and computes the start and end indexes 
    
    Args:
        x: (list) input signal
        frame_size: (int) frame size for the analysis in Saturation Detector
        hop_size: (int) hop_size for the analysis in Saturation Detector
    
    Kwargs:

    Returns:
        starts: start indexes
        ends: end indexes
        count: number of frames saturated
    """
    saturationDetector = estd.SaturationDetector(frameSize=frame_size, hopSize=hop_size, **kwargs)

    ends = []
    starts = []
    count = 0
    for frame in estd.FrameGenerator(x, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        frame_starts, frame_ends = saturationDetector(frame)

        for s in frame_starts:
            starts.append(s)
        for e in frame_ends:
            ends.append(e)
        if len(e) + len(s) == 0: count += 1

    return starts, ends, count