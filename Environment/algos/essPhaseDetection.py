from essentia.standard import FalseStereoDetector, StereoDemuxer, FrameGenerator, StereoMuxer


def essFalsestereoDetector(x: list, frameSize=1024, hopSize=512, correlationthreshold=0.98, percentageThreshold=90):
    """Computes the correlation and consideres if the information in the two channels is the same
    
    Args:
        x: (list) input signal
        frameSize: (int) frame size for the analysis in falseStereoDetector
        hopSize: (int) hop_size for the analysis in falseStereoDetector
        correlationthreshold: (float) lower limit to decide if a file has correlation problems

    Returns:
        final_bool: (bool) True if the information is the same in both channels, False otherwise
        percentace: (float) How many frames were false stereo over all the frames
    """
    rx, lx = StereoDemuxer()(x)

    lfg = FrameGenerator(lx, frameSize=frameSize, hopSize=hopSize, startFromZero=True)
    rfg = FrameGenerator(rx, frameSize=frameSize, hopSize=hopSize, startFromZero=True)

    mux = StereoMuxer()

    total = 0
    count = 0

    falseStereoDetector = FalseStereoDetector()

    for frameL, frameR in zip(lfg, rfg):
        print(falseStereoDetector(mux(frameL, frameR)))
        if falseStereoDetector(mux(frameL, frameR))[1] > correlationthreshold:
            count += 1
        # frame_bool, _ = estd.FalseStereoDetector()(frame)
        # if frame_bool == 1: count += 1
        total += 1

    falseStereoDetector.reset()
    percentage = 100*count/total
    
    return round(percentage, 2), percentage > percentageThreshold


def outofPhaseDetector(x: list, frameSize=1024, hopSize=512, correlationthreshold=-0.8, percentageThreshold=90):
    """Computes the correlation and flags the file if the file has a 90% of frames out of phase
    
    Args:
        x: (list) input signal
        frameSize: (int) frame size for the analysis in falseStereoDetector
        hopSize: (int) hop_size for the analysis in falseStereoDetector
        correlationthreshold: (float) higher limit to decide if a file has correlation problems

    Returns:
        final_bool: (bool) True if the information is the same in both channels, False otherwise
        percentace: (float) How many frames were false stereo over all the frames
    """
    rx, lx = StereoDemuxer()(x)

    lfg = FrameGenerator(lx, frameSize=frameSize, hopSize=hopSize, startFromZero=True)
    rfg = FrameGenerator(rx, frameSize=frameSize, hopSize=hopSize, startFromZero=True)

    mux = StereoMuxer()

    total = 0
    count = 0
    falseStereoDetector = FalseStereoDetector()

    for frameL, frameR in zip(lfg, rfg):
        print(falseStereoDetector(mux(frameL, frameR))[1])
        if falseStereoDetector(mux(frameL, frameR))[1] < correlationthreshold:
            count += 1
        total += 1

    falseStereoDetector.reset()
    percentage = 100*count/total

    return round(percentage, 2), percentage > percentageThreshold
