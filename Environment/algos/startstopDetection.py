from essentia.standard import StartStopCut
from essentia import array as esarr


def essStartstopDetector(x, frameSize=1024, hopSize=512, **kwargs):
    """Breaks x into frames and computes the start and end indexes.
    
    Args:
        x: (list) input signal
        frameSize: (int) frame size for the analysis in StartStopCut
        hopSize: (int) hopSize for the analysis in StartStopCut
    
    Kwargs:
        same **kwargs for StartStopCut

    Returns:
        ratio of the startcut + stopcut vs the whole audio length
    """

    startStopCut = StartStopCut(frameSize=frameSize, hopSize=hopSize, **kwargs)

    startCut, stopCut = startStopCut(esarr(x))

    # len_x = len(x)
    # del x; del startStopCut;
    return round(100*(startCut + stopCut)/len(x), 2)
