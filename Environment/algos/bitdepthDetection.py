import numpy as np


def convert_to_bin_array(value:int, b:int):
    return [int(val) for val in np.binary_repr(value, width=b)]
    # if value > 2**(b-1)-1: raise ValueError("Value too large")
    # if value < -2**(b-1): raise ValueError("Value too small")
    # bin_str = np.binary_repr(value, width = b)
    # return [int(val) for val in bin_str]


def randomize_chunks(audio: list, chunk_len: int, number_of_chunks: int):
    random_audio_splices = []
    for _ in range(number_of_chunks):
        start_idx = np.random.randint(0, len(audio-chunk_len-1))
        random_audio_splices = [*random_audio_splices, *audio[int(start_idx):int(start_idx+chunk_len)]]
    return random_audio_splices


def bit_depth_detector(audio: list, b: int, chunk_len=100, number_of_chunks=100):

    if audio.shape[1] > 1:
        audio = audio[:, 0]

    if b not in [8, 16, 24, 32]:
        raise ValueError("Only bit depths accepted are 8, 16, 24, 32")

    # set audio to be ints from -2**(b-1) to 2**(b-1)-1
    # and change data type according to the bit depth of the container
    audio = (2**(b-1)) * audio.astype('float64')

    if b == 8:
        audio = audio.astype('int8')
    elif b == 16:
        audio = audio.astype('int16')
    elif b == 24:
        audio = audio.astype('int32')
    elif b == 32:
        audio = audio.astype('int32')
    else:
        raise ValueError("Only bit depths accepted are 8, 16, 24, 32")

    # clip the values to the maximum values represented by the datatype selected
    audio = np.clip(audio, -2**(b-1), 2**(b-1)-1)

    # get number_of_chunks random splices of data of chunk_len samples each one
    audio_random = randomize_chunks(audio, int(chunk_len), int(number_of_chunks))

    # initialise logic array to b positions of 0
    result = [0]*b

    for sample in audio_random:
        bin_arr = convert_to_bin_array(sample, b)  # compute the binary number for each sample

        # if any bit is used once, it is converted to 1, only bits that remain unused
        # through all the samples are kept as 0
        result = [a or b for a, b in zip(result, bin_arr)]

    for i, el in enumerate(reversed(result)):
        if el != 0:
            # the smallest position with a 0 will determine the number of unused bits
            return len(result)-i, (len(result)-i) < b
    return 0, True
