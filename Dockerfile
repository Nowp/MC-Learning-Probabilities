FROM python:3.10.4-bullseye

RUN apt update

# Install storm
## Install storm requirements
RUN apt install -y openjdk-17-jre build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev

## Build storm
WORKDIR /opt
RUN git clone https://github.com/moves-rwth/storm.git --branch 1.6.4
WORKDIR /opt/storm/build
RUN cmake -DSTORM_USE_SPOT_SYSTEM=OFF -DSTORM_USE_SPOT_SHIPPED=ON ..
RUN make

# Install stormpy
## Install stormpy requirements
RUN apt install maven -y
WORKDIR /opt
RUN git clone https://github.com/ths-rwth/carl-parser --branch master14
WORKDIR /opt/carl-parser/build
RUN cmake ..
RUN make 

WORKDIR /opt
RUN git clone https://github.com/moves-rwth/pycarl.git --branch 2.0.5
WORKDIR /opt/pycarl/
RUN python3 setup.py build_ext --jobs 1 develop

## Build stormpy
WORKDIR /opt
RUN git clone https://github.com/moves-rwth/stormpy.git
WORKDIR /opt/stormpy
RUN git reset --hard 2c9dca4
RUN python3 setup.py build_ext --jobs 1 develop ; exit 0

WORKDIR /home/stormpy-dev