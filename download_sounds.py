import freesound, sys,os
from pydub import AudioSegment
import subprocess
import argparse
import shutil

def downloadSounds(API_key,outputDir,queryText,topNResults,duration,tag):

    c = freesound.FreesoundClient()
    c.set_token(API_key,"oauth")

    if type(duration) == tuple:
        flt_dur = " duration:[" + str(duration[0])+ " TO " +str(duration[1]) + "]"
    else:
        flt_dur = ""

    if tag != "":
        flt_tag = "tag:"+tag
    else:
        flt_tag = ""

    page_size = 30
    if not flt_tag + flt_dur == "":
        results = c.text_search(query=queryText ,filter = flt_tag + flt_dur,sort="score", fields="type,id,name,previews,username,url,analysis", page_size=page_size, normalized=1)
    else:
        results = c.text_search(query=queryText ,sort="score",fields="type,id,name,previews,username,url,analysis", page_size=page_size, normalized=1)
    
    pageNo = 1
    indCnt = 0
    sndCnt = 0
    totalSnds = min(results.count,200)
    print(results.count)

    wavDir = outputDir + "/wav"
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    if not os.path.exists(wavDir):
        os.mkdir(wavDir)

    downloadedSounds = []

    while(1):
        if indCnt >= totalSnds:
            print("Not able to download required number of sounds. Either there are not enough search results on freesound for your search query and filtering constraints or something is wrong with this script.")
            break
        sound = results[indCnt - ((pageNo-1)*page_size)]

        soundname = "{0}{1}{2}".format(str(sound.id),".",sound.type)
        outfilenamepath = os.path.join(outputDir,soundname)
        if not os.path.exists(outfilenamepath):
            sound.retrieve(outputDir,soundname)
        print(soundname)
        if not sound.type=="wav":
            AudioSegment.from_file(outfilenamepath,sound.type).export("{0}{1}{2}{3}".format(wavDir,"/",str(sound.id),".wav"),"wav")
        else:
            shutil.copyfile(outfilenamepath,"{0}{1}{2}".format(wavDir,"/",soundname))
        downloadedSounds.append([sound.name, str(sound.id), sound.url])
        sndCnt+=1

        indCnt+=1

        if indCnt%page_size==0:
            results = results.next_page()
            pageNo+=1

        if sndCnt>=topNResults:
            break

    print("Number of sounds downloaded: " + str(sndCnt))
    fid = open(os.path.join(outputDir, "{0}{1}".format(queryText,'_SoundList.txt')), 'w')
    for elem in downloadedSounds:
        fid.write('\t'.join(elem)+'\n')
    fid.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate correlation for all the sounds in s folder")
    parser.add_argument("--API_key", help="API key for freesound.org", default="R1PKtXHKnGGIZD6KvfFCmuJaJASOBb",required=False)
    parser.add_argument("--outputDir", help="where will the mp3 files be stored. Wav conversions will be stored in outputDir/wav/", default="",required=False)
    parser.add_argument("--queryText", help="search query", default="",required=False)
    parser.add_argument("--topNResults", help="max number of sounds to be downloaded (the number will be capped at 200)",default=10,required=False)
    parser.add_argument("--duration", help="tuple for the range of length in seconds for the search. Don't care = \"\"",default=(0,5),required=False)
    parser.add_argument("--tag", help="tags to search in the query",default="",required=False)
    args = parser.parse_args()
    downloadSounds(API_key=args.API_key,outputDir=args.outputDir,queryText=args.queryText,topNResults=int(args.topNResults),duration=args.duration,tag=args.tag)
