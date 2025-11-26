FROM python:3.12-slim

WORKDIR /train

COPY . /train
    
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    cmake \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install -r requirements.txt

COPY . .

# ENV port=8080

# EXPOSE 8080

CMD ["python3","train.py"]