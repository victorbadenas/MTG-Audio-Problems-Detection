import numpy as np

def pearson_calculation(frame:list):
    """Calculates the pearson coefficient
	Args:
		frame: (np.array) containing the frame of an audio signal

	Returns:
		correlation: (float)
		boolean: containing True if the frame should be skipped in further calculations and False otherwise

	"""
    frame = np.array(frame)
    if frame.shape[1] == 1:
        return 1, False

    cov = np.cov(frame[:,0],frame[:,1])[0][1]
    stdL = np.std(frame[:,0])
    stdR = np.std(frame[:,1])
    if(stdR*stdL == 0):
        return 0, True
    else:
        return np.clip(cov/(stdR*stdL),-1,1), False

def falsestereo_detector(x:list, frame_size=1024, hop_size=512):
    """Computes the correlation and consideres if the information in the two channels is the same
    
    Args:
        x: (list) input signal

    Returns:
        final_bool: (bool) True if the information is the same in both channels, False otherwise
        percentace: (float) How many frames were false stereo over all the frames
    """
    #print(x.shape[1])
    if x.shape[1] == 1: return True,100.00
    
    frame_size = int(frame_size)
    hop_size = int(hop_size)

    N_frames = int( np.ceil( (int(x.shape[0]) - frame_size) / hop_size ))
    zp = np.zeros(( int(N_frames * hop_size + frame_size) - len(x), 2), dtype=x.dtype)
    x = np.concatenate((x, zp))
    #print(x.shape)
    count = 0
    total = 0
    for idx in range(N_frames):
        frame = x[idx:idx+frame_size,:]
        corr, skip_bool = pearson_calculation(frame)
        if corr >= 0.98 and not skip_bool: count += 1
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

    if x.shape[1] == 1: return True,100.00
    
    frame_size = int(frame_size)
    hop_size = int(hop_size)

    N_frames = int( np.ceil( (int(x.shape[0]) - frame_size) / hop_size ))
    zp = np.zeros(( int(N_frames * hop_size + frame_size) - len(x), 2), dtype=x.dtype)
    x = np.concatenate((x, zp))
    #print(x.shape)
    count = 0
    total = 0
    for idx in range(N_frames):
        frame = x[idx:idx+frame_size,:]
        corr, skip_bool = pearson_calculation(frame)
        if corr <= -0.8 and not skip_bool: count += 1
        total += 1
    
    percentage = 100*count/total
    return percentage > 90.0, round(percentage,2)