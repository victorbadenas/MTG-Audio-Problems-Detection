FROM ubuntu

RUN apt-get update
RUN apt-get install -y git-core
RUN mkdir git/

RUN apt install -y python3.7
RUN apt install -y python3-pip
RUN pip3 install 'setuptools<20.2'

RUN mkdir /data
COPY python_requirements.txt /
RUN pip3 install -r python_requirements.txt

RUN git clone https://github.com/MTG/freesound-python
RUN git clone https://github.com/MTG/essentia.git

RUN cd /freesound-python && python3 setup.py install
RUN cd /essentia && python3 setup.py install

WORKDIR /data

COPY . .