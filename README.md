# MTG-Audio-Problems-Detection
Master Thesis 

This is the master branch for the audio problems detection master thesis

docker image build -t mtg_audio_problems_detection:1.0 .

docker run -v $(pwd):/data/ -p 8080:8080 -it mtg_audio_problems_detection:1.0 bash