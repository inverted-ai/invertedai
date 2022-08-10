FROM carlasim/carla
# FROM ubuntu
USER root
# NVIDIA has changed CUDA Linux GPG Repository Keys
# Must delete the old keys and add new ones
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update -y
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH
# Install Python and related tooling
# RUN apt-get install -y libffi-dev python python-dev python3-dev python3-pip python3-venv \
#   && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
#   && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
#   && pip install setuptools distro wheel \
  # && apt-get install -y build-essential llvm libxml2-dev



# RUN apt-get update && apt-get install -y python3-pip
# RUN pip3 install jupyterlab
RUN pip3 install notebook carla
# RUN pip3 install numpy

USER carla
WORKDIR /home/carla/
# RUN apt-get install -y \
#   python3

# CMD jupyter notebook
# CMD sh
CMD ["/bin/bash", "./CarlaUE4.sh", "-fps=10"]
