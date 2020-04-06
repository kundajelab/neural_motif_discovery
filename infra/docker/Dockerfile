# Use the official TensorFlow image as parent
FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Set the working directory
WORKDIR /scratch

# Copy the set of requirements for training
COPY train_requirements.txt /scratch

# Install any needed packages specified in requirements.txt
RUN pip install -r train_requirements.txt

# Install DeepSHAP
ADD shap /scratch/shap
RUN pip install -e shap

# Create /users/amtseng directory
RUN mkdir -p /users/amtseng

# Set environment variables for Python
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
