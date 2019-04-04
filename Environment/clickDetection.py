import essentia.standard as estd

def ess_click_detector(x, frame_size=1024, hop_size=512, **kwargs):
    """Breaks x into frames and computes the start and end indexes.
    
    Args:
        x: (list) input signal
        frame_size: (int) frame size for the analysis in Click Detector
        hop_size: (int) hop_size for the analysis in Click Detector
    
    Kwargs:
        same **kwargs for ClickDetector

    Returns:
        starts: start indexes
        ends: end indexes
        percentage of frames with the issue
    """

    clickDetector = estd.ClickDetector(frameSize=frame_size, hopSize=hop_size, **kwargs)

    ends = []
    starts = []
    count = 0
    
    for i,frame in enumerate(estd.FrameGenerator(x, frameSize=frame_size, hopSize=hop_size, startFromZero=True)):
        frame_starts, frame_ends = clickDetector(frame)

        for s in frame_starts:
            starts.append(s)
        for e in frame_ends:
            ends.append(e)
        
        if len(frame_starts) + len(frame_ends) != 0: count += 1
    
    print("Number of frames:", i+1)
    return starts, ends, round(100*count/(i+1),2)