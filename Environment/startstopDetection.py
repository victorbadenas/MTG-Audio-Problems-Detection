import essentia.standard as estd
from essentia import array as esarr

def ess_startstopdetector(x, frame_size=1024, hop_size=512, **kwargs):
    """Breaks x into frames and computes the start and end indexes.
    
    Args:
        x: (list) input signal
        frame_size: (int) frame size for the analysis in StartStopCut
        hop_size: (int) hop_size for the analysis in StartStopCut
    
    Kwargs:
        same **kwargs for StartStopCut

    Returns:
        ratio of the startcut + stopcut vs the whole audio length
    """

    startStopCut = estd.StartStopCut(frameSize=frame_size, hopSize=hop_size, **kwargs)

    startCut, stopCut = startStopCut(esarr(x))

    return 100 * (startCut + stopCut) / len(x)