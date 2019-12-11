FROM python:3.7-slim-buster

# Install 
RUN apt-get update && apt-get install -y python3-dev build-essential

# Make directories suited to your application
RUN mkdir -p /home/dss-street-vendor
WORKDIR /home/dss-street-vendor

# Install pip
RUN pip3 install --upgrade pip

# Copy and install requirements
COPY requirements.txt /home/dss-street-vendor
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy contents from your local to your docker container
COPY . /home/dss-street-vendor
COPY ./notebooks/ /notebooks
COPY ./app /app 

# Running
CMD ["python3","-m","uvicorn","--host","0.0.0.0","--port","5000","main:app"]

