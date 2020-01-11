FROM python:3.7
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
RUN mkdir /code/matlab
WORKDIR /code
RUN mkdir -p ~/data_files
RUN mkdir -p ~/static
COPY requirements.txt /code/
COPY matlab/ /code/matlab
RUN ls -la /code/matlab
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install -y octave
RUN apt-get install -y octave-statistics

COPY . /code/
