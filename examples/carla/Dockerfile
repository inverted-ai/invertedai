FROM carlasim/carla:0.9.13

USER root
WORKDIR /home/carla/Import
RUN apt install -y wget
RUN wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.13.tar.gz
WORKDIR /home/carla
RUN ./ImportAssets.sh
USER carla