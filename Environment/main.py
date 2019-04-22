import os
import argparse
from pydub import AudioSegment
from single_file_json_compute import single_json_compute

def multi_file_compute(audiofolder, jsonfolder):
    """Calls the audio_problems_detection algorithms and stores the result in a json file

    Args:
        audiofolder: string containing the relative path for the folder containing the audio files
        jsonfolder: string containing the relative path for the folder containing the json files

    """
    first = True #boolean to create the headers of the problems
    tsvpath = os.path.join(jsonfolder,"all_files.tsv")
    if os.path.exists(tsvpath): os.remove(tsvpath)
    
    for file in os.listdir(audiofolder):

        filename, extension = os.path.splitext(file)

        #if the file is mp3 of aiff will be converted to wav and the mp3 or aiff file will be removed
        if extension in (".mp3",".aiff"):
            print("{} is {} and will be converted to wav".format(file,extension))
            audiopath_src = os.path.join(audiofolder, file)
            audiopath_dest = os.path.join(audiofolder, filename + ".wav")
            AudioSegment.from_file(audiopath_src).export(audiopath_dest, "wav")
            os.remove(audiopath_src)
            extension = ".wav"

        if extension == ".wav":
            audiopath = os.path.join(audiofolder,file)
            json_dict = single_json_compute(audiopath, jsonfolder, print_flag = False)
    
            with open(tsvpath, 'a') as tsvfile:
                if first: #this will write the headers when the loop is in the first audio file
                    tsvfile.write("Filename")
                    for problem in json_dict:
                        for feature in json_dict[problem]:
                            tsvfile.write("\t" + problem + ":" + feature)
                    tsvfile.write("\n")
                    first = False
                
                tsvfile.write(filename)
                for problem in json_dict:
                    for feature in json_dict[problem]:
                         tsvfile.write("\t" + str(json_dict[problem][feature]))
                tsvfile.write("\n")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Calls the audio_problems_detection algorithms and stores the result in a json file")
    parser.add_argument("audiofolder", help="relative path to the audiofile")
    parser.add_argument("--jsonfolder", help="string containing the relative path for the json file", default="",required=False)
    args = parser.parse_args()

    if args.jsonfolder == "":
        jsonfolder = os.path.join(args.audiofolder, "json")
        if not os.path.exists(jsonfolder):
            os.mkdir(jsonfolder)
        print("json folder is:", jsonfolder)
        multi_file_compute(args.audiofolder,jsonfolder)
    else:
        multi_file_compute(args.audiofolder,args.jsonfolder)
