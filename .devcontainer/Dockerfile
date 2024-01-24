FROM python:3.9


RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

RUN mkdir -m 700 /root/.ssh; \
  touch -m 600 /root/.ssh/known_hosts; \
  ssh-keyscan github.com > /root/.ssh/known_hosts
RUN cd /home && git clone https://github.com/AprilRobotics/apriltag.git 
WORKDIR /home/apriltag

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 cmake -y
RUN cd /home/apriltag && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --target install

RUN  pip3 install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org dt-apriltags
RUN  pip3 install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org opencv-python

RUN mkdir /home/app
WORKDIR /home/app
RUN alias python="python3"


