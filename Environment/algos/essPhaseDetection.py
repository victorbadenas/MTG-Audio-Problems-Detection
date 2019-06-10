from essentia.standard import FalseStereoDetector, StereoDemuxer, FrameGenerator, StereoMuxer, AudioLoader


def essFalsestereoDetector(x: list, frameSize=1024, hopSize=512, correlationThreshold=0.98, percentageThreshold=90, channels=2, **kwargs):
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
    if channels < 2:
        return 1, False, True

    rx, lx = StereoDemuxer()(x)
    mux = StereoMuxer()
    falseStereoDetector = FalseStereoDetector(correlationThreshold=correlationThreshold, **kwargs)

    lfg = FrameGenerator(lx, frameSize=frameSize, hopSize=hopSize, startFromZero=True)
    rfg = FrameGenerator(rx, frameSize=frameSize, hopSize=hopSize, startFromZero=True)

    problematicFrames = sum([falseStereoDetector(mux(frameL, frameR))[0] for frameL, frameR in zip(lfg, rfg)])
    # problematicFrames = []
    # for frameL, frameR in zip(lfg, rfg):
    #     res, corr = falseStereoDetector(mux(frameL, frameR))
    #     problematicFrames.append(res)

    falseStereoDetector.reset()

    conf = float(sum(problematicFrames)) / float(lfg.num_frames())

    return conf, conf > percentageThreshold/100, False


def outofPhaseDetector(x: list, frameSize=1024, hopSize=512, correlationThreshold=-0.8, percentageThreshold=90, channels=2, **kwargs):
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
    if channels < 2:
        return 1, False, True

    rx, lx = StereoDemuxer()(x)
    mux = StereoMuxer()
    falseStereoDetector = FalseStereoDetector(**kwargs)

    lfg = FrameGenerator(lx, frameSize=frameSize, hopSize=hopSize, startFromZero=True)
    rfg = FrameGenerator(rx, frameSize=frameSize, hopSize=hopSize, startFromZero=True)

    problematicFrames = 0
    for frameL, frameR in zip(lfg, rfg):
        _, corr = falseStereoDetector(mux(frameL, frameR))
        problematicFrames += corr < correlationThreshold
    falseStereoDetector.reset()

    conf = problematicFrames / lfg.num_frames()

    return conf, conf > percentageThreshold/100, False
