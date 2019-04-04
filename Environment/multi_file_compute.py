import os
import json
import shutil
import pandas as pd
from pydub import AudioSegment
from single_file_json_compute import *


def multi_file_compute(audiofolder, jsonfolder):
    """Calls the audio_problems_detection algorithms and stores the result in a json file

    Args:
        audiofolder: string containing the relative path for the folder containing the audio files
        jsonfolder: string containing the relative path for the folder containing the json files

    """
    df = pd.DataFrame()
    for file in os.listdir(audiofolder):
        filename, extension = os.path.splitext(file)
        if extension in (".mp3",".aiff"):
            print("mp3 or aiff")
            audiopath_src = os.path.join(audiofolder, file)
            audiopath_dest = os.path.join(audiofolder, filename + ".wav")
            AudioSegment.from_file(audiopath_src).export(audiopath_dest, "wav")
            os.remove(audiopath_src)
            extension = ".wav"

        if extension == ".wav":
            audiopath = os.path.join(audiofolder,file)
            single_json_compute(audiopath, jsonfolder, print_flag = False)

            jsonpath = os.path.join(jsonfolder,filename + ".json")
            with open(jsonpath,'r') as jsonfile:
                json_dict = json.load(jsonfile)
            
            df_dict = { "Name" : filename }
            for problem in json_dict:
                for feature in json_dict[problem]:
                    name = problem + ':' + feature
                    df_dict[name] = [json_dict[problem][feature]]

            df_file = pd.DataFrame(df_dict)
            df = df.append(df_file)
    df = df.set_index("Name")

    with open(jsonfolder+"all_files.tsv",'w') as tsvfile:
        df.to_csv(tsvfile, sep="\t")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Calls the audio_problems_detection algorithms and stores the result in a json file")
    parser.add_argument("audiofolder", help="relative path to the audiofile")
    parser.add_argument("--jsonfolder", help="string containing the relative path for the json file", default="",required=False)
    args = parser.parse_args()

    if args.jsonfolder == "":
        print("json folder is:", args.audiofolder)
        multi_file_compute(args.audiofolder,args.audiofolder)
    else:
        multi_file_compute(args.audiofolder,args.jsonfolder)
