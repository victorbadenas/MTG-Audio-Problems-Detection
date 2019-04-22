import essentia.standard as estd

def ess_saturation_detector(x:list, frame_size=1024, hop_size=512, **kwargs):
    """Breaks x into frames and computes the start and end indexes 
    
    Args:
        x: (list) input signal
        frame_size: (int) frame size for the analysis in Saturation Detector
        hop_size: (int) hop_size for the analysis in Saturation Detector
    
    Kwargs:
        Same **kwargs than the ones for SaturationDetector

    Returns:
        starts: start indexes
        ends: end indexes
        percentage of frames with the issue
    """
    saturationDetector = estd.SaturationDetector(frameSize=frame_size, hopSize=hop_size, **kwargs)

    ends = []
    starts = []
    count = 0
    for i,frame in enumerate(estd.FrameGenerator(x, frameSize=frame_size, hopSize=hop_size, startFromZero=True)):
        frame_starts, frame_ends = saturationDetector(frame)

        for s in frame_starts:
            starts.append(s)
        for e in frame_ends:
            ends.append(e)
        if len(frame_starts) + len(frame_ends) != 0: count += 1

    #del frame; del frame_starts; del frame_ends; del x
    #x = None; frame = None; frame_starts = None; frame_ends= None
    return starts, ends, round(100*count/(i+1),2)