FROM nvidia/cuda:10.2-devel-centos7
MAINTAINER VISHNU BALAKRISHNAN

# Update and install necessary packages
RUN yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y \
    vim \
    git \
    wget \
    ca-certificates

# Install EPEL repository and IUS repository for newer Python versions
RUN yum install -y epel-release && \
    yum install -y https://repo.ius.io/ius-release-el7.rpm && \
    yum update -y

# Install Python 2.7, Python 3.6, Python 3.8, and their respective pip and python-dev packages
RUN yum install -y \
    python2 python2-pip python2-devel \
    python36 python36-pip python36-devel \
    python38 python38-pip python38-devel

# Upgrade pip for Python 2.7
RUN pip2 install --upgrade "pip < 21.0"

# Install compatible numpy, matplotlib for Python 2.7
RUN pip2 install "numpy < 1.17" "matplotlib < 2.2.5" --user

# Install required libraries for Pillow
RUN yum install -y \
    libtiff-devel \
    libjpeg-devel \
    openjpeg2-devel \
    zlib-devel \
    freetype-devel \
    lcms2-devel \
    libwebp-devel \
    tcl-devel \
    tk-devel \
    harfbuzz-devel \
    fribidi-devel \
    libraqm-devel

# Install numpy, matplotlib for Python 3.6 and 3.8
RUN pip3.6 install numpy matplotlib --user
RUN pip3 install numpy matplotlib --user

# Install packages for Python 3.6 pip
RUN pip3.6 install --user matplotlib pandas emcee corner "schwimmbad < 0.3.0" sympy scipy && \
    pip3.6 install --user scikit-monaco


RUN git clone https://github.com/vishnubk/dedisp.git && \
   cd dedisp &&\
   git checkout arch61 && \
   make clean && \
   make -j 32 && \
   make install 

RUN git clone https://github.com/vishnubk/3D_peasoup.git && \
   cd 3D_peasoup && \
   make clean && \
   make -j 32 && \
   make install 
  
RUN ldconfig /usr/local/lib


RUN yum install -y fftw fftw-devel fftw-libs-single
RUN pip2 install --user --index-url=https://pypi.org/simple/ numpy

#Sigpyproc
RUN git clone https://github.com/ewanbarr/sigpyproc.git
# sigpyproc
ENV SIGPYPROC="/sigpyproc" \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/sigpyproc/lib/c"
WORKDIR sigpyproc
RUN python2 setup.py install --record list.txt --user


#5D Peasoup
RUN git clone https://github.com/vishnubk/5D_Peasoup.git && \
    cd 5D_Peasoup && \
    git checkout fast_bt_resampler_working && \
    make clean && \
    make -j 32 && \
    make install


RUN ldconfig /usr/local/lib




WORKDIR /home



