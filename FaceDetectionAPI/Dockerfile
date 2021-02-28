FROM nvcr.io/nvidia/pytorch:20.03-py3

ENV NVIDIA_VISIBLE_DEVICES all

RUN DEBIAN_FRONTEND=noninteractive apt-get update

RUN apt-get update && \
    apt-get install -y libpq-dev && \
    apt-get install -y build-essential && \
    apt-get install -y python3-all-dev && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y libxml2-dev

RUN mkdir -p /workspace

WORKDIR /workspace

# uwsgi
#RUN pip install uwsgi

COPY ./requirements.txt /workspace
COPY ./entrypoint.sh /workspace
COPY . /workspace

RUN pip install -r requirements.txt

RUN pwd
RUN ls

ENTRYPOINT ["/workspace/entrypoint.sh"]
#/FaceDetectionAPI/face_detect_api/face_detect_api