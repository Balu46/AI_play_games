FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /train

COPY . /train

RUN pip install -r requirements.txt

COPY . .

ENV port=8080

EXPOSE 8080

CMD ["python3","train.py"]