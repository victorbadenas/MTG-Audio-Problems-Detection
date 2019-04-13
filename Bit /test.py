import essentia.standard as estd

def Bit_Detection(directory:str):



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate correlation for all the sounds in s folder")
	parser.add_argument("directory", help="Directory of the files")
	args = parser.parse_args()
	Bit_Detection(args.directory)