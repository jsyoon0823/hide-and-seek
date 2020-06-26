FROM nvidia/cuda:10.2-base

RUN apt-get update -qy
RUN apt-get install -qy python3-pip python3-numpy
RUN pip3 install --upgrade pip
RUN pip3 install docker tensorflow-gpu==1.15.2 pandas tqdm scikit-learn argparse keras==2.3.1
