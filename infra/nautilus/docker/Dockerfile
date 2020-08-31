# Use the official TensorFlow image as parent
FROM tensorflow/tensorflow:1.14.0-gpu

# Set the working directory
WORKDIR /scratch

# Install some basic utilities
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda with Python 3.7 into /opt
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy

# Enable Conda and alter bashrc so the Conda default environment is always activated
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Attach Conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Copy the set of requirements for training
COPY train_requirements.txt /scratch

# Install any needed packages specified in requirements.txt
RUN pip install -r train_requirements.txt

# Install DeepSHAP
RUN git clone https://github.com/atseng95/shap.git
RUN pip install -e shap

# Create /users/amtseng directory
RUN mkdir -p /users/amtseng

# Set environment variables for Python
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
