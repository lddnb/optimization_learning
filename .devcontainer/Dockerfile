FROM osrf/ros:jazzy-desktop-noble

RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt-get clean
RUN apt-get -y update --fix-missing

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

ENV HTTP_PROXY http://127.0.0.1:7890
ENV HTTPS_PROXY http://127.0.0.1:7890

# setup timezone
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata \
    locales \
    software-properties-common && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# install packages
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    build-essential \
    cmake \
    clang \
    clangd \
    clang-format \
    curl \
    git \
    gnupg \
    pciutils \
    iputils-ping \
    iproute2 \
    libblosc-dev \
    libjemalloc-dev \
    libclang-dev \
    libc++-dev \
    libpoco-dev \
    lsb-release \
    tar \
    zip \
    unzip \
    tmux \
    vim \
    wget \
    zsh \
    sudo \
    xz-utils && \
    add-apt-repository universe

RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    libgflags-dev \
    libunwind-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libceres-dev \
    libfmt-dev \
    ros-jazzy-desktop-full \
    ros-jazzy-gtsam

# 清除apt缓存
RUN apt autoremove -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'root:      ' | chpasswd
RUN echo 'ubuntu:      ' | chpasswd
USER ubuntu
RUN mkdir -p /home/ubuntu/ros_ws/src
WORKDIR "/home/ubuntu/ros_ws"

RUN git config --global http.proxy http://127.0.0.1:7890 && \
    git config --global https.proxy http://127.0.0.1:7890

RUN wget -e "https_proxy=http://127.0.0.1:7890" https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | sh && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    sed -i '/^plugins=/c\plugins=(git zsh-autosuggestions zsh-syntax-highlighting z)' ~/.zshrc

RUN echo 'ENABLE_CORRECTION="true"' >> ~/.zshrc
RUN echo 'COMPLETION_WAITING_DOTS="true"' >> ~/.zshrc
RUN echo 'alias sss="source devel/setup.zsh"' >> ~/.zshrc
RUN echo 'alias sis="source install/setup.zsh"' >> ~/.zshrc
RUN echo source "/opt/ros/jazzy/setup.zsh" >> ~/.zshrc
