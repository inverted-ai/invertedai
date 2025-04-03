FROM ubuntu:22.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    && apt-get install -y build-essential \
    && apt-get install -y gcc-10 g++-10 cpp-10 gdb\
    && apt-get -y install libopencv-dev \
    && apt-get -y install libboost-all-dev \
    && apt-get install -y apt-transport-https curl gnupg \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg \
    && mv bazel-archive-keyring.gpg /usr/share/keyrings \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] \
    https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
    && apt-get -y update && apt-get -y install bazel=7.6.1

COPY . /usr/src/myapp
WORKDIR /usr/src/myapp

RUN chmod 755 ./run_all_tests.sh

CMD ["/bin/bash"]
