FROM nvidia/cuda:12.4.1-devel-ubuntu22.04


RUN apt update
# python | pip | git to clone | curl to dl nvm | libglib2.0 is libgthread | mesa-utils is opengl for opencv
RUN apt install -y python3 python3-pip git curl libglib2.0-0 mesa-utils

WORKDIR /app
RUN git clone https://github.com/cinnamon-bootcamp-wukong/final-project/
COPY ./run.sh final-project/
RUN chmod +x final-project/run.sh

ENV NVM_DIR /root/.nvm
ENV NODE_VERSION 19.9.0

# install node
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install $NODE_VERSION && \
    nvm use $NODE_VERSION && \
    nvm alias default $NODE_VERSION

# Add nvm to path
ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# Verify node and npm are installed correctly
RUN node -v && npm -v

# dependencies
WORKDIR /app/final-project
RUN pip install -r requirements.txt
RUN pip install opencv-python
WORKDIR /app/final-project/frontend
RUN npm install next


# checkpoint
WORKDIR /app/final-project
RUN gdown 17xve1HRBAiDACOviOmGQbFNE8u2z4ZVi -O backend/checkpoints/

# lfg
ENTRYPOINT [ "./run.sh" ]