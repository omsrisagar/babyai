# To build this docker image:
# docker build -t babyai_kg .
#
# To run the image:
# docker run -it babyai_kg
#FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
#FROM nvidia/11.6.0-runtime-ubuntu18.04
#FROM nvidia/cuda:11.3.1-base-ubuntu18.04
FROM nvidia/cuda:11.1.1-base-ubuntu18.04
CMD nvidia-smi

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install -y unzip
RUN apt-get install -y qt5-default qttools5-dev-tools git
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
#RUN adduser --disabled-password --gecos '' --shell /bin/bash vsadhu \
# && chown -R vsadhu:vsadhu /app
#RUN echo "vsadhu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-vsadhu
#RUN chmod +x /app
#USER vsadhu

# All users can use /home/user as their home directory
#ENV HOME=/home/vsadhu
#RUN chmod 777 /home/vsadhu

# this will reset the working directory to /
FROM python:3.7
RUN pip install --upgrade pip

# Install Miniconda and Python 3.8
#ENV CONDA_AUTO_UPDATE_CONDA=false
#ENV PATH=/home/vsadhu/miniconda/bin:$PATH
#RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
# && chmod +x ~/miniconda.sh \
# && ~/miniconda.sh -b -p ~/miniconda \
# && rm ~/miniconda.sh \
# && conda install -y python==3.8.1 \
# && conda clean -ya

# CUDA 10.2-specific steps --> needed is started from base image
# RUN conda install -y -c pytorch \
#    cudatoolkit=10.2 \
#    "pytorch=1.5.0=py3.8_cuda10.2.89_cudnn7.6.5_0" \
#    "torchvision=0.6.0=py38_cu102" \
# && conda clean -ya
#RUN conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# Create a working directory
WORKDIR /app
#RUN ls -la /app
#RUN pwd
ADD requirements.txt .
RUN pip install -r requirements.txt
# [No need now] Need to install gym_minigrid after installing babyai as babyai installs default gym_minigrid (without my modifications)
RUN pip install gym_minigrid -e git+https://github.com/omsrisagar/gym-minigrid.git@master#egg=gym_minigrid
#RUN pip install jericho -e git+https://github.com/microsoft/jericho.git@6f761073ef064e62412c36cc8de569f57b39561c#egg=jericho

RUN pwd
RUN ls -la .

# ADD babyai . will copy all contnents of babyai into current wdir. It will not copy the folder as a whole!
ADD setup.py .
ADD babyai ./babyai
ADD scripts ./scripts

# Good cmds to debug
#RUN pwd
#RUN ls -la .

RUN pip install --editable .

#RUN pip install babyai -e git+https://github.com/omsrisagar/babyai.git@kg#egg=babyai

#CMD ["pwd"]
#RUN ls -la /babyai_kg/*

# Set the working directory to babayai/scripts
WORKDIR /app/scripts/
#RUN chown /app
#RUN ls -la .
# Launch the run command
#CMD ["python", "train_rl.py", "--model debug_graph", "--procs 2", "--val-episodes 1"]
#CMD python train_rl.py --model debug_graph --procs 2 --val-episodes 1
