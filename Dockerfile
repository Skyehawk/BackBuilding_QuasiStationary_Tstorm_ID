FROM python:3.8-slim-buster

# Install python and any other necessary dependencies
# awscli is necessary for the run_job.sh script to access S3 resources

copy requirements.txt requirements.txt
run pip3 install -r requirements.txt

# Copy the local folder to the Docker image
COPY ./ /usr/local/scripts

# Set the working directory to the newly created folder
WORKDIR /usr/local/scripts
