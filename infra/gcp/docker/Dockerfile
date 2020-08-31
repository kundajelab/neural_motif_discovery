# Use a simple Python3 base image
FROM python:3

# Set the working directory
WORKDIR /scratch

# Install LS-GKM, extract binaries to /opt/lsgkm/bin/
RUN cd /opt/ && \
	wget https://github.com/kundajelab/lsgkm/archive/master.zip && \
	unzip master.zip && \
	mv lsgkm-master lsgkm && \
	cd lsgkm/src && \
	make

# Install Google Cloud SDK
RUN apt-get update && apt install -y --allow-unauthenticated wget
RUN cd /opt/ && \
	wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-307.0.0-linux-x86_64.tar.gz && \
	tar xvfz google-cloud-sdk-307.0.0-linux-x86_64.tar.gz && \
	./google-cloud-sdk/install.sh
ENV PATH "$PATH:/opt/google-cloud-sdk/bin/"

# Create /users/amtseng directory
RUN mkdir -p /users/amtseng

# Set environment variables for Python
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
