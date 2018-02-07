FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
LABEL maintainer="Kazuhiro Ota <zektbach@gmail.com>"

WORKDIR /workspace

RUN apt update -y && apt upgrade -y
RUN apt install -y python3-dev python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
ADD requirements_mac.txt .
RUN pip3 install -r requirements_mac.txt

CMD ["python3", "train.py"]
