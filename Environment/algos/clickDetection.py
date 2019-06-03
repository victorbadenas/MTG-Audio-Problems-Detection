from essentia.standard import ClickDetector, FrameGenerator


def essClickDetector(x, frameSize=1024, hopSize=512, **kwargs):
    """Breaks x into frames and computes the start and end indexes.
    
    Args:
        x: (list) input signal
        frameSize: (int) frame size for the analysis in Click Detector
        hopSize: (int) hopSize for the analysis in Click Detector
    
    Kwargs:
        same **kwargs for ClickDetector

    Returns:
        starts: start indexes
        ends: end indexes
        percentage of frames with the issue
    """

    clickDetector = ClickDetector(frameSize=frameSize, hopSize=hopSize, **kwargs)

    ends = []
    starts = []
    count = 0
    total = 0

    for frame in FrameGenerator(x, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        frame_starts, frame_ends = clickDetector(frame)

        for s in frame_starts:
            starts.append(s)
        for e in frame_ends:
            ends.append(e)
        
        if len(frame_starts) + len(frame_ends) != 0:
            count += 1
        total += 1
    # print("Number of frames:", i+1)
    # del x; del frame; del frame_ends; del frame_starts;
    return starts, ends, round(100*count/total, 2)
