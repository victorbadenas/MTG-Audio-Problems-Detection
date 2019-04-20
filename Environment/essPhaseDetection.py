import essentia.standard as estd

def falsestereo_detector(x:list, frame_size=1024, hop_size=512):
    """Computes the correlation and consideres if the information in the two channels is the same
    
    Args:
        x: (list) input signal

    Returns:
        final_bool: (bool) True if the information is the same in both channels, False otherwise
        percentace: (float) How many frames were false stereo over all the frames
    """
    Lx, Rx = estd.StereoDemuxer()(x)

    LFG = estd.FrameGenerator(Lx, frameSize=frame_size, hopSize=hop_size, startFromZero=True)
    RFG = estd.FrameGenerator(Rx, frameSize=frame_size, hopSize=hop_size, startFromZero=True)

    mux = estd.StereoMuxer()

    total = 0
    count = 0
    for frameL,frameR in zip(LFG,RFG):
        if estd.FalseStereoDetector()(mux(frameL,frameR))[0] == 1: count += 1
        #frame_bool, _ = estd.FalseStereoDetector()(frame)
        #if frame_bool == 1: count += 1
        total += 1
    
    percentage = 100*count/total
    
    return percentage > 90.0, round(percentage,2)

def outofphase_detector(x:list, frame_size=1024, hop_size=512):
    """Computes the correlation and flags the file if the file has a 90% of frames out of phase
    
    Args:
        x: (list) input signal

    Returns:
        final_bool: (bool) True if the information is the same in both channels, False otherwise
        percentace: (float) How many frames were false stereo over all the frames
    """
    Lx, Rx = estd.StereoDemuxer()(x)

    LFG = estd.FrameGenerator(Lx, frameSize=frame_size, hopSize=hop_size, startFromZero=True)
    RFG = estd.FrameGenerator(Rx, frameSize=frame_size, hopSize=hop_size, startFromZero=True)

    mux = estd.StereoMuxer()

    total = 0
    count = 0
    for frameL,frameR in zip(LFG,RFG):
        if estd.FalseStereoDetector()(mux(frameL,frameR))[1] < -0.8: count += 1
        total += 1
    
    percentage = 100*count/total
    final_bool = percentage > 90.0
    
    return final_bool, round(percentage,2)